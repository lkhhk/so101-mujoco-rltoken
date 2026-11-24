"""Train matched B1 RLT actor-critics with and without relabeled demonstrations."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEMO = ROOT / "rlt-stage2-demo-replay-b1-20261122"
sys.path.insert(0, str(DEMO))
import convert_scripted_replay as source

placement = source.placement
grasp = source.grasp
WARMUP_SEEDS = list(range(54000, 54006))
ONLINE_SEEDS = list(range(54100, 54108))
EVAL_SEEDS = list(range(55000, 55006))
EVAL_REPEATS = 2
WARMUP_UPDATES = 300
UPDATES_PER_EPISODE = 80
ARMS = ("vla_warmup", "demo_warmup")


def atomic_torch(path, data):
    pending = path.with_suffix(".tmp")
    torch.save(data, pending)
    pending.replace(path)


@torch.no_grad()
def relabel_one(env, mapping, frozen, entry, row):
    seed = entry["seed"]
    env.reset(seed=seed)
    tracker = grasp.Stages(env)
    handoff_step = row["handoff_step"]
    with np.load(grasp.DATA / grasp_directory(entry["episode"]) / "motion.npz") as motion:
        for command in motion["action"][:handoff_step]:
            target = np.clip(mapping.radians(command), env.low, env.high)
            _, _, success, timeout, _ = env.step(env.action_for_targets(target))
            assert not success and not timeout
            tracker.update(False)
    assert env.steps == handoff_step and tracker.grasp == row["stable_lift_step"]
    handoff = placement.stats_at_boundary(env, tracker)
    post_handoff = placement.Handoff(handoff, "stable_lift")
    continuation_seed = seed + 70000
    torch.manual_seed(continuation_seed)
    ended = timeout = False
    while not (ended or timeout):
        actions, _ = frozen.observe(env, mapping, with_token=False)
        commands = frozen.post(actions[None]).float().cpu().numpy()[0]
        for command in commands:
            target = np.clip(mapping.radians(command), env.low, env.high)
            _, _, ended, timeout, _ = env.step(env.action_for_targets(target))
            tracker.update(ended)
            post_handoff.update(env, tracker)
            if ended or timeout:
                break
    replay = list(entry["replay"])
    tail = replay[-1]
    assert tail[4].item() == 1 and tail[5].item() == 0
    replay[-1] = (*tail[:4], torch.tensor([float(ended)]), *tail[5:])
    source.review(replay, expect_terminal=bool(ended))
    return {**entry, "replay": replay}, {
        "seed": seed, "episode": entry["episode"], "continuation_seed": continuation_seed,
        "success": bool(ended), "steps": env.steps, "handoff": vars(post_handoff),
        "stable_placement_step": tracker.placed,
    }


def grasp_directory(episode):
    manifest = getattr(grasp_directory, "rows", None)
    if manifest is None:
        rows = json.loads((grasp.DATA / "manifest.json").read_text())["episodes"]
        manifest = {row["episode"]: row["directory"] for row in rows}
        grasp_directory.rows = manifest
    return manifest[episode]


class TrialLearner(grasp.Learner):
    def __init__(self):
        super().__init__()
        self.is_demo = []
        self.actor_updates = 0

    def add(self, rows, is_demo=False):
        self.replay.extend(rows)
        self.is_demo.extend([is_demo] * len(rows))

    def fit(self, steps, train_actor):
        assert len(self.replay) >= 32 and len(self.is_demo) == len(self.replay)
        for _ in range(steps):
            indices = self.prng.choices(range(len(self.replay)), k=64)
            rows = [self.replay[index] for index in indices]
            batch = [torch.stack(values).to("cuda") for values in zip(*rows)]
            z, state, ref, action, reward, discount, next_z, next_state, next_ref = batch
            with torch.no_grad():
                next_mean, next_std = self.actor_target(next_z, next_state, next_ref)
                noise = torch.randn(next_std.shape, device="cuda", generator=self.noise) * next_std
                next_action = grasp.constrained(next_mean + noise, next_ref)
                target = (reward + discount * self.critic_target.q_min(next_z, next_state, next_action)).clamp(0, 1)
            q1, q2 = self.critic(z, state, action)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            self.step_optimizer(critic_loss, self.critic, self.critic_optim, 10)
            self.updates += 1
            self.latest["critic_loss"] = critic_loss.item()
            if train_actor and self.updates % 2 == 0:
                demo = torch.tensor([self.is_demo[index] for index in indices], device="cuda")
                reachable = ((action - ref).abs() <= grasp.ACTION_DELTA + 1e-6).all(dim=(1, 2))
                imitate = demo & reachable
                anchor_target = torch.where(imitate[:, None, None], action, ref)
                drop = torch.rand((len(z), 1, 1), device="cuda", generator=self.noise) < 0.5
                mean, std = self.actor(z, state, torch.where(drop, 0.0, ref))
                noise = torch.randn(std.shape, device="cuda", generator=self.noise) * std
                proposal = grasp.constrained(mean + noise, ref)
                self.critic.requires_grad_(False)
                try:
                    actor_q = self.critic.q_min(z, state, proposal).mean()
                    anchor_loss = F.mse_loss(mean, anchor_target)
                    self.step_optimizer(-actor_q + grasp.BETA * anchor_loss, self.actor, self.actor_optim, 1)
                finally:
                    self.critic.requires_grad_(True)
                self.actor_updates += 1
                self.latest.update({"actor_q": actor_q.item(), "anchor_mse": anchor_loss.item(),
                                    "demo_imitation_samples": int(imitate.sum().item())})
            with torch.no_grad():
                for source, target_model in ((self.critic, self.critic_target), (self.actor, self.actor_target)):
                    for param, target_param in zip(source.parameters(), target_model.parameters()):
                        target_param.lerp_(param, 0.005)

    @staticmethod
    def step_optimizer(loss, model, optimizer, norm):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), norm)
        if not torch.isfinite(loss) or not torch.isfinite(grad):
            raise FloatingPointError("Non-finite actor or critic loss/gradient")
        optimizer.step()

    def status(self):
        return {**super().status(), "actor_updates": self.actor_updates, "demo_chunks": sum(self.is_demo)}


def save_agent(path, learner, label, episode, frozen):
    atomic_torch(path, {
        "schema_version": 1, "arm": label, "online_episodes": episode, "config": vars(learner.config),
        "actor_state_dict": learner.actor.state_dict(), "critic_state_dict": learner.critic.state_dict(),
        "actor_target_state_dict": learner.actor_target.state_dict(),
        "critic_target_state_dict": learner.critic_target.state_dict(),
        "actor_optimizer_state_dict": learner.actor_optim.state_dict(),
        "critic_optimizer_state_dict": learner.critic_optim.state_dict(),
        "learner": learner.status(), "replay": learner.replay, "is_demo": learner.is_demo,
        "rng": {"sampler": learner.prng.getstate(), "noise": learner.noise.get_state()},
        "reward": "stable_placement_after_frozen_vla_handoff", "vla_provenance": frozen.provenance,
        "rl_token_checkpoint": str(grasp.STAGE1 / "stage1.pt"), "policy_batch_size": 1,
        "chunk_steps": grasp.CHUNK, "action_delta_bound": grasp.ACTION_DELTA,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relabel-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    declared = WARMUP_SEEDS + ONLINE_SEEDS + EVAL_SEEDS
    assert len(declared) == len(set(declared))
    manifest = json.loads((grasp.DATA / "manifest.json").read_text())
    assert not set(declared) & {row["seed"] for row in manifest["episodes"]}
    source_rows = json.loads((DEMO / "report.json").read_text())
    assert source_rows["complete"]
    demos = torch.load(DEMO / "scripted-replay.pt", map_location="cpu", weights_only=True)
    previous_reference = torch.load(DEMO / "reference-repeat-replay.pt", map_location="cpu", weights_only=True)
    assert demos["fields"] == source.FIELDS == previous_reference["fields"]
    assert len(demos["episodes"]) == 24
    env = grasp.SO101BlockBox(ROOT / "menagerie")
    mapping = grasp.JointMapping(json.loads((ROOT / "official_block_follower.json").read_text()))
    began = perf_counter()
    report = {
        "complete": False, "batch_size": 1, "relabel": [], "warmup": [],
        "online": {label: [] for label in ARMS}, "evaluate": {},
        "settings": {"warmup_seeds": WARMUP_SEEDS, "online_seeds": ONLINE_SEEDS, "eval_seeds": EVAL_SEEDS,
            "eval_repeats": EVAL_REPEATS, "warmup_critic_updates": WARMUP_UPDATES,
            "critic_updates_per_online_episode": UPDATES_PER_EPISODE,
            "reference_dropout": 0.5, "chunk_steps": grasp.CHUNK, "action_delta_bound": grasp.ACTION_DELTA},
    }
    report_path = HERE / "report.json"
    relabeled_path = HERE / "relabeled-demo-replay.pt"

    def progress(stage, row, learner=None):
        grasp.save(report_path, report)
        brief = {"stage": stage, "seed": row["seed"], "success": row["success"], "steps": row["steps"]}
        if learner:
            brief.update(learner.status())
        print(json.dumps(brief), flush=True)

    try:
        frozen = grasp.FrozenPolicy()
        expected_vla = demos["vla_provenance"]
        assert frozen.provenance["policy_version"] == expected_vla["policy_version"]
        if relabeled_path.exists():
            cache = torch.load(relabeled_path, map_location="cpu", weights_only=True)
            assert cache["policy_version"] == expected_vla["policy_version"] and cache["fields"] == source.FIELDS
            relabeled, report["relabel"] = cache["episodes"], cache["report"]
            assert len(relabeled) == len(report["relabel"]) == 24
        else:
            relabeled = []
            for entry, row in zip(demos["episodes"], source_rows["episodes"]):
                assert entry["seed"] == row["seed"]
                converted, metrics = relabel_one(env, mapping, frozen, entry, row)
                relabeled.append(converted)
                report["relabel"].append(metrics)
                progress("relabel", metrics)
            atomic_torch(relabeled_path, {"schema_version": 1, "fields": source.FIELDS,
                         "policy_version": expected_vla["policy_version"],
                         "reward": "frozen_vla_continuation_placement",
                         "episodes": relabeled, "report": report["relabel"]})
        demo_chunks = [chunk for entry in relabeled for chunk in entry["replay"]]
        assert len(demo_chunks) == 168
        report["relabel_summary"] = {
            "episodes": len(relabeled), "chunks": len(demo_chunks),
            "positive_terminals": sum(row[4].item() > 0 for row in demo_chunks),
        }
        grasp.save(report_path, report)
        print(json.dumps({"relabel_summary": report["relabel_summary"]}), flush=True)
        if args.relabel_only:
            return
        reference_rows = [row for entry in previous_reference["episodes"] for row in entry["replay"]]
        for seed in WARMUP_SEEDS:
            row, replay = placement.rollout(env, mapping, frozen, seed, learn=True)
            reference_rows.extend(replay)
            report["warmup"].append(row)
            progress("warmup", row)
        agents = {label: TrialLearner() for label in ARMS}
        for key, value in agents[ARMS[0]].actor.state_dict().items():
            assert torch.equal(value, agents[ARMS[1]].actor.state_dict()[key])
        for key, value in agents[ARMS[0]].critic.state_dict().items():
            assert torch.equal(value, agents[ARMS[1]].critic.state_dict()[key])
        for label, learner in agents.items():
            learner.add(reference_rows)
            if label == "demo_warmup":
                learner.add(demo_chunks, is_demo=True)
            learner.fit(WARMUP_UPDATES, train_actor=False)
        report["initial_pools"] = {label: learner.status() for label, learner in agents.items()}
        report["checkpoints"] = {}
        for episode, seed in enumerate(ONLINE_SEEDS, 1):
            for label in ARMS:
                learner = agents[label]
                row, replay = placement.rollout(env, mapping, frozen, seed, learner.actor, learn=True)
                learner.add(replay)
                learner.fit(UPDATES_PER_EPISODE, train_actor=True)
                report["online"][label].append(row)
                if episode in (len(ONLINE_SEEDS) // 2, len(ONLINE_SEEDS)):
                    checkpoint = HERE / f"{label}-episode-{episode:02d}.pt"
                    save_agent(checkpoint, learner, label, episode, frozen)
                    report["checkpoints"][label] = str(checkpoint)
                progress("online_" + label, row, learner)
        report["final_learners"] = {label: learner.status() for label, learner in agents.items()}
        assert all(value["critic_updates"] == WARMUP_UPDATES + UPDATES_PER_EPISODE * len(ONLINE_SEEDS)
                   and value["actor_updates"] == UPDATES_PER_EPISODE * len(ONLINE_SEEDS) // 2
                   for value in report["final_learners"].values())
        models = {"vla": None}
        for label in ARMS:
            models[label] = placement.retrieve_actor(report["checkpoints"][label], agents[label].config)
        report["evaluate"] = {label: [] for label in models}
        for repeat in range(1, EVAL_REPEATS + 1):
            labels = tuple(models) if repeat % 2 else tuple(reversed(models))
            for seed in EVAL_SEEDS:
                for label in labels:
                    row, _ = placement.rollout(env, mapping, frozen, seed, models[label])
                    row["repeat"] = repeat
                    report["evaluate"][label].append(row)
                    progress("eval_" + label, row)
        report["evaluate_summary"] = {label: placement.summary(rows) for label, rows in report["evaluate"].items()}
        report["contrasts"] = {
            "demo_vs_vla": placement.contrasts(report["evaluate"]["vla"], report["evaluate"]["demo_warmup"]),
            "demo_vs_no_demo": placement.contrasts(report["evaluate"]["vla_warmup"], report["evaluate"]["demo_warmup"]),
        }
        report["seconds"] = round(perf_counter() - began, 1)
        report["complete"] = True
        grasp.save(report_path, report)
        print(json.dumps({key: report[key] for key in (
            "relabel_summary", "final_learners", "evaluate_summary", "contrasts", "seconds"
        )}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
