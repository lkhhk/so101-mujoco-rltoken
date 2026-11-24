"""MuJoCo Stage 2 pilot. Frozen PI05 and RLT at B1, grasp-only actor control."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np
from safetensors.torch import load_file
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGE1 = ROOT / "rlt-stage1-b16-b1-20261122"
BASELINE = ROOT / "rlt-b1-baseline-20261122"
sys.path[:0] = [str(STAGE1), str(BASELINE)]
from validate_stage1_b1 import encode, load_encoder
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTChunkActor, RLTTwinCritic
from train_stage1_cache import PI05RLTConfig
from measure_stages import ADAPTER, ADAPTER_SHA, BASE, BASE_SHA, CAMERAS, DATA, TOKENIZER
from measure_stages import JointMapping, SO101BlockBox, Stages, save
from evaluate_so101_mujoco_lora import REFERENCE, TASK
from probe_rlt_batches import load_ira_lora, make_processors, predict_with_features

CHUNK = 10
GATE_RADIUS = 0.040
PHASE_STEPS = 240
ACTION_DELTA = 0.35
GAMMA = 0.99
BETA = 1.0
UTD = 5
WARMUP_EPISODES = 12
ONLINE_EPISODES = 16
EVAL_EPISODES = 16
SEED = 20261122


def gate(env, tracker):
    return np.linalg.norm(env.data.site("gripperframe").xpos - tracker.position()) <= GATE_RADIUS


def control_check(env, mapping, episode):
    env.reset(seed=episode["seed"])
    tracker = Stages(env)
    gate_step = None
    with np.load(DATA / episode["directory"] / "motion.npz") as motion:
        actions = motion["action"]
    for command in actions:
        if env.steps % CHUNK == 0 and gate_step is None and gate(env, tracker):
            gate_step = env.steps
        targets = np.clip(mapping.radians(command), env.low, env.high)
        _, _, success, _, _ = env.step(env.action_for_targets(targets))
        tracker.update(success)
        if success:
            break
    assert gate_step is not None and gate_step < tracker.first_contact < tracker.grasp < tracker.placed
    return {"seed": episode["seed"], "gate_step": gate_step, **tracker.report()}


class FrozenPolicy:
    def __init__(self):
        self.encoder, self.encoder_step = load_encoder()
        self.policy, self.provenance = load_ira_lora(BASE, ADAPTER, device="cuda")
        assert self.provenance["base_sha256"] == BASE_SHA and self.provenance["adapter_sha256"] == ADAPTER_SHA
        self.policy.requires_grad_(False).eval()
        self.pre, self.post = make_processors(self.policy, ADAPTER, TOKENIZER, "cuda")
        stats = load_file(str(ADAPTER / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
        self.action_mean = stats["action.mean"].to("cuda")
        self.action_scale = stats["action.std"].to("cuda") + 1e-8
        self.state_mean = stats["observation.state.mean"].to("cuda")
        self.state_scale = stats["observation.state.std"].to("cuda") + 1e-8
        known = torch.linspace(-1, 1, CHUNK * 6, device="cuda").view(1, CHUNK, 6)
        torch.testing.assert_close(self.post(known).cpu(), self.physical(known).cpu(), atol=1e-5, rtol=1e-5)

    def physical(self, actions):
        return actions * self.action_scale + self.action_mean

    def normalized(self, physical):
        return (physical - self.action_mean) / self.action_scale

    def observe(self, env, mapping, with_token):
        raw = torch.from_numpy(mapping.observation(env.data.qpos[env.position]))
        obs = {"observation.state": raw, "task": TASK}
        for name in CAMERAS:
            obs[f"observation.images.{name}"] = (
                torch.from_numpy(env.render_camera(name)).permute(2, 0, 1).float() / 255
            )
        batch = self.pre(obs)
        if with_token:
            core = self.policy.get_base_model().model
            noise = core.sample_noise((1, 50, self.policy.config.max_action_dim), device="cuda")
            actions, features, mask = predict_with_features(self.policy, batch, noise)
            token = encode(self.encoder, features, mask)[0]
            state = ((raw.to("cuda") - self.state_mean) / self.state_scale).clamp(-10, 10)
            return actions[:, :CHUNK].float()[0], (token, state)
        actions = self.policy.predict_action_chunk(batch)
        return actions[:, :CHUNK].float()[0], None


def constrained(proposal, reference):
    return reference + (proposal - reference).clamp(-ACTION_DELTA, ACTION_DELTA)


def split_identity_initialization(actor):
    # Route signed reference coordinates through the upstream two-layer ReLU MLP.
    # This starts the full-output actor at the teacher action with trainable features.
    hidden, _, output = actor.net
    dimensions = CHUNK * 6
    eye = torch.eye(dimensions, device=hidden.weight.device)
    with torch.no_grad():
        hidden.weight[:2 * dimensions].zero_()
        hidden.bias[:2 * dimensions].zero_()
        hidden.weight[:dimensions, -dimensions:] = eye
        hidden.weight[dimensions:2 * dimensions, -dimensions:] = -eye
        output.weight.zero_()
        output.bias.zero_()
        output.weight[:, :dimensions] = eye
        output.weight[:, dimensions:2 * dimensions] = -eye
        zero = torch.zeros(1, 2048, device=hidden.weight.device)
        state = torch.zeros(1, 6, device=hidden.weight.device)
        reference = torch.linspace(-1, 1, dimensions, device=hidden.weight.device).view(1, CHUNK, 6)
        torch.testing.assert_close(actor(zero, state, reference)[0], reference, atol=0, rtol=0)


class Learner:
    def __init__(self):
        torch.manual_seed(SEED)
        self.prng = random.Random(SEED)
        self.noise = torch.Generator(device="cuda").manual_seed(SEED + 1)
        self.config = PI05RLTConfig(mode="online_rl", bc_weight=BETA, discount=GAMMA)
        self.actor = RLTChunkActor(self.config).cuda()
        split_identity_initialization(self.actor)
        self.critic = RLTTwinCritic(self.config).cuda()
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=1e-4)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=3e-4)
        self.replay = []
        self.updates = 0
        self.latest = {}

    def update(self, new_rows, train_actor=True, limit=None):
        self.replay.extend(new_rows)
        if len(self.replay) < 32:
            return
        count = min(len(new_rows) * UTD, limit) if limit is not None else len(new_rows) * UTD
        for _ in range(count):
            indices = self.prng.choices(range(len(self.replay)), k=64)
            fields = [self.replay[index] for index in indices]
            batch = [torch.stack(values).to("cuda") for values in zip(*fields)]
            z, state, ref, action, reward, discount, next_z, next_state, next_ref = batch
            with torch.no_grad():
                next_mean, next_std = self.actor_target(next_z, next_state, next_ref)
                noise = torch.randn(next_std.shape, device="cuda", generator=self.noise) * next_std
                next_action = constrained(next_mean + noise, next_ref)
                target = reward + discount * self.critic_target.q_min(next_z, next_state, next_action)
                target = target.clamp(0, 1)
            q1, q2 = self.critic(z, state, action)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            self.critic_optim.zero_grad(set_to_none=True)
            critic_loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
            if not torch.isfinite(grad) or not torch.isfinite(critic_loss):
                raise FloatingPointError("Non-finite critic update")
            self.critic_optim.step()
            self.updates += 1
            self.latest["critic_loss"] = critic_loss.item()
            if train_actor and self.updates % 2 == 0:
                drop = torch.rand((len(z), 1, 1), device="cuda", generator=self.noise) < 0.5
                mean, std = self.actor(z, state, torch.where(drop, 0.0, ref))
                noise = torch.randn(std.shape, device="cuda", generator=self.noise) * std
                sampled = constrained(mean + noise, ref)
                self.critic.requires_grad_(False)
                try:
                    actor_q = self.critic.q_min(z, state, sampled).mean()
                    anchor = F.mse_loss(mean, ref)
                    actor_loss = -actor_q + BETA * anchor
                    self.actor_optim.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                    if not torch.isfinite(grad) or not torch.isfinite(actor_loss):
                        raise FloatingPointError("Non-finite actor update")
                    self.actor_optim.step()
                finally:
                    self.critic.requires_grad_(True)
                self.latest.update({"actor_loss": actor_loss.item(), "actor_q": actor_q.item(),
                                    "reference_mse": anchor.item()})
            with torch.no_grad():
                for source, target_net in ((self.critic, self.critic_target), (self.actor, self.actor_target)):
                    for weight, target_weight in zip(source.parameters(), target_net.parameters()):
                        target_weight.lerp_(weight, 0.005)

    def status(self):
        positives = sum(row[4].item() > 0 for row in self.replay)
        return {"replay_chunks": len(self.replay), "positive_terminal_chunks": positives,
                "critic_updates": self.updates, **self.latest}


def store_transition(snapshot, action, reward, discount, next_snapshot):
    z, state, ref = snapshot
    if next_snapshot is None:
        next_snapshot = tuple(torch.zeros_like(item) for item in snapshot)
    return tuple(item.detach().cpu() for item in (z, state, ref, action)) + (
        torch.tensor([reward]), torch.tensor([discount]),
    ) + tuple(item.detach().cpu() for item in next_snapshot)


@torch.no_grad()
def rollout(env, mapping, frozen, learner, seed, mode):
    torch.manual_seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed + SEED)
    env.reset(seed=seed)
    tracker = Stages(env)
    collect = mode in ("warmup", "online")
    window_start = hand_back = None
    phase_result = None
    replay = []
    pending = None
    actor_chunks = 0
    scales = []
    terminated = timeout = False
    begin = perf_counter()
    while not (terminated or timeout):
        if window_start is None and tracker.grasp is None and gate(env, tracker):
            window_start = env.steps
        active = window_start is not None and phase_result is None
        ref, inputs = frozen.observe(env, mapping, with_token=active)
        action = ref
        snapshot = None
        if active:
            z, state = inputs
            snapshot = (z, state, ref)
            if pending is not None:
                replay.append(store_transition(*pending, snapshot))
                pending = None
            if mode in ("online", "eval"):
                proposal, std = learner.actor(z, state, ref)
                if mode == "online":
                    proposal = proposal + torch.randn(std.shape, device="cuda", generator=generator) * std
                action = constrained(proposal[0], ref)
                scales.append(float((action - ref).square().mean().sqrt()))
                actor_chunks += 1
        physical = frozen.post(action[None]).float().numpy()[0]
        requested = mapping.radians(physical)
        targets = np.clip(requested, env.low, env.high)
        if active and not np.array_equal(targets, requested):
            actual = (targets - mapping.home) / mapping.scale + REFERENCE
            action = frozen.normalized(torch.as_tensor(actual, dtype=torch.float32, device="cuda"))
        discounted_reward = 0.0
        for index, target in enumerate(targets):
            previous = tracker.grasp
            _, _, terminated, timeout, _ = env.step(env.action_for_targets(target))
            tracker.update(terminated)
            if active and previous is None and tracker.grasp is not None:
                discounted_reward = GAMMA ** index
            if terminated or timeout:
                break
        if active and (tracker.grasp is not None or env.steps - window_start >= PHASE_STEPS or timeout or terminated):
            phase_result = "stable_lift" if tracker.grasp is not None else "phase_timeout"
            hand_back = env.steps
            if collect:
                replay.append(store_transition(snapshot, action, discounted_reward, 0.0, None))
                break
        elif active and collect:
            pending = (snapshot, action, 0.0, GAMMA ** CHUNK)
    assert pending is None
    return {
        **tracker.report(), "seed": seed, "mode": mode, "steps": env.steps,
        "success": tracker.placed is not None, "gate_step": window_start, "hand_back_step": hand_back,
        "phase_result": phase_result or "gate_not_reached", "actor_chunks": actor_chunks,
        "mean_reference_deviation": float(np.mean(scales)) if scales else None,
        "wall_seconds": round(perf_counter() - begin, 2),
    }, replay


def paired(baseline, evaluated):
    refs = {row["seed"]: row for row in baseline["episodes"]}
    pairs = [(refs[row["seed"]], row) for row in evaluated]
    assert len(pairs) == EVAL_EPISODES
    contrasts = {}
    for label, outcome in (
        ("stable_lift", lambda row: row["stable_lift_step"] is not None),
        ("placement", lambda row: row["success"]),
    ):
        before = [outcome(ref) for ref, _ in pairs]
        after = [outcome(row) for _, row in pairs]
        contrasts[label] = {
            "baseline": sum(before), "actor": sum(after),
            "actor_only": sum(b and not a for a, b in zip(before, after)),
            "baseline_only": sum(a and not b for a, b in zip(before, after)),
        }
    return contrasts


def persist(learner):
    path = HERE / "grasp-actor-critic.pt"
    staging = path.with_suffix(".tmp")
    torch.save({
        "schema_version": 1, "config": vars(learner.config),
        "actor_state_dict": learner.actor.state_dict(), "critic_state_dict": learner.critic.state_dict(),
        "actor_target_state_dict": learner.actor_target.state_dict(),
        "critic_target_state_dict": learner.critic_target.state_dict(),
        "actor_optimizer_state_dict": learner.actor_optim.state_dict(),
        "critic_optimizer_state_dict": learner.critic_optim.state_dict(),
        "learner": learner.status(), "stage1": str(STAGE1 / "stage1.pt"),
        "replay": learner.replay, "policy_batch_size": 1, "gate_radius_meters": GATE_RADIUS,
        "control_chunk_steps": CHUNK, "phase_steps": PHASE_STEPS, "action_delta_bound": ACTION_DELTA,
    }, staging)
    staging.replace(path)
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    manifest = json.loads((DATA / "manifest.json").read_text())
    baseline = json.loads((BASELINE / "c10-n30.json").read_text())
    evaluation_seeds = baseline["spec"]["seeds"][:EVAL_EPISODES]
    warmup_seeds = list(range(41000, 41000 + WARMUP_EPISODES))
    train_seeds = list(range(41000 + WARMUP_EPISODES, 41000 + WARMUP_EPISODES + ONLINE_EPISODES))
    assert baseline["complete"] and baseline["spec"]["execute_steps"] == CHUNK
    assert not set(warmup_seeds + train_seeds) & set(evaluation_seeds)
    assert not set(warmup_seeds + train_seeds) & {row["seed"] for row in manifest["episodes"]}
    env = SO101BlockBox(ROOT / "menagerie")
    mapping = JointMapping(json.loads((ROOT / "official_block_follower.json").read_text()))
    try:
        controls = [control_check(env, mapping, row) for row in manifest["episodes"][:3]]
        print(json.dumps({"gate_controls": controls}), flush=True)
        frozen = FrozenPolicy()
        learner = Learner()
        begin = perf_counter()
        report = {
            "complete": False, "settings": {"batch_size": 1, "chunk": CHUNK,
                "gate_radius_meters": GATE_RADIUS, "phase_steps": PHASE_STEPS,
                "action_delta_bound": ACTION_DELTA, "warmup_seeds": warmup_seeds,
                "online_seeds": train_seeds, "eval_seeds": evaluation_seeds},
            "gate_controls": controls, "stage1_step": frozen.encoder_step,
            "vla_provenance": frozen.provenance, "warmup": [], "online": [], "baseline": [], "eval": [],
        }
        if args.smoke:
            row, replay = rollout(env, mapping, frozen, learner, 41000, "warmup")
            assert row["gate_step"] is not None and replay
            # Duplicate rows only for a shape and numerical check in this disposable run.
            learner.update(replay * (32 // len(replay) + 1), limit=2)
            assert learner.updates == 2 and "actor_loss" in learner.latest
            print(json.dumps({"smoke": row, "learner": learner.status()}), flush=True)
            return
        path = HERE / "report.json"
        for mode, seeds in (("warmup", warmup_seeds), ("online", train_seeds),
                            ("baseline", evaluation_seeds), ("eval", evaluation_seeds)):
            for seed in seeds:
                row, replay = rollout(env, mapping, frozen, learner, seed, mode)
                if mode in ("warmup", "online"):
                    learner.update(replay, train_actor=(mode == "online"))
                report[mode].append(row)
                report["learner"] = learner.status()
                save(path, report)
                print(json.dumps({**row, "learner": report["learner"]}), flush=True)
            if mode == "warmup":
                assert len(learner.replay) >= 32 and learner.status()["positive_terminal_chunks"] >= 2
            if mode == "online":
                report["checkpoint"] = persist(learner)
                save(path, report)
        report["comparison"] = paired({"episodes": report["baseline"]}, report["eval"])
        report["complete"] = True
        report["seconds"] = round(perf_counter() - begin, 1)
        save(path, report)
        print(json.dumps({"comparison": report["comparison"], "learner": report["learner"],
                          "checkpoint": report["checkpoint"], "seconds": report["seconds"]}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
