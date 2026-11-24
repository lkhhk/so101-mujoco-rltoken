"""Measure grasp handoffs and train a grasp actor using the full placement outcome."""
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PILOT = ROOT / "rlt-stage2-grasp-b1-20261122"
sys.path.insert(0, str(PILOT))
import train_grasp_rlt as grasp

WARMUP = list(range(52000, 52008))
ONLINE = list(range(52008, 52020))
DEVELOPMENT = list(range(50000, 50006))
TEST = list(range(51000, 51012))
CHECKPOINT_STEPS = (6, 12)
FAILURE_BAND_MM = 8.0


def stats_at_boundary(env, tracker):
    inputs = env.data.site("gripperframe")
    cube = tracker.position()
    box = env.data.mocap_pos[env.box_mocap]
    contacts = set()
    for contact in env.data.contact:
        first, second = contact.geom
        if env.cube_geom in (first, second):
            other = second if first == env.cube_geom else first
            contacts.add(env.model.body(env.model.geom_bodyid[other]).name)
    local = inputs.xmat.reshape(3, 3).T @ (cube - inputs.xpos)
    return {
        "step": env.steps, "cube_xyz_m": cube.tolist(),
        "cube_orientation_wxyz": env.data.qpos[env.cube_pos + 3:env.cube_pos + 7].tolist(),
        "cube_height_gain_mm": float((cube[2] - tracker.initial_z) * 1000),
        "cube_in_gripper_frame_mm": (local * 1000).tolist(),
        "cube_to_box_xy_mm": float(np.linalg.norm(cube[:2] - box[:2]) * 1000),
        "cube_speed_m_per_s": float(np.linalg.norm(env.data.qvel[env.cube_vel:env.cube_vel + 3])),
        "gripper_joint_rad": float(env.data.qpos[env.position[-1]]),
        "arm_joint_rad": env.data.qpos[env.position[:5]].tolist(),
        "contact_bodies": sorted(contacts),
        "held_steps": tracker.hold_count, "bilateral_held_steps": tracker.bilateral_count,
    }


@dataclass
class Handoff:
    snapshot: dict
    result: str
    early_trace: list = field(default_factory=list)
    first_drop_step: int | None = None

    def update(self, env, tracker):
        age = env.steps - self.snapshot["step"]
        if 0 < age <= 60 and age % 5 == 0:
            self.early_trace.append(stats_at_boundary(env, tracker))
        if self.result == "stable_lift" and self.first_drop_step is None:
            cube = tracker.position()
            box = env.data.mocap_pos[env.box_mocap]
            outside_box = np.any(np.abs(cube[:2] - box[:2]) > np.array([0.055, 0.050]))
            if outside_box and (cube[2] - tracker.initial_z) * 1000 <= FAILURE_BAND_MM:
                self.first_drop_step = env.steps


@torch.no_grad()
def rollout(env, mapping, frozen, seed, actor=None, learn=False):
    torch.manual_seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed + grasp.SEED)
    env.reset(seed=seed)
    tracker = grasp.Stages(env)
    start = None
    transfer = None
    replay = []
    previous = None
    terminal = None
    acted = 0
    ended = timeout = False
    begin = perf_counter()
    while not (ended or timeout):
        if start is None and tracker.grasp is None and grasp.gate(env, tracker):
            start = stats_at_boundary(env, tracker)
        active = start is not None and transfer is None
        ref, inputs = frozen.observe(env, mapping, with_token=active)
        action = ref
        if active:
            z, state = inputs
            snapshot = (z, state, ref)
            if previous is not None:
                replay.append(grasp.store_transition(*previous, snapshot))
                previous = None
            if actor is not None:
                mean, std = actor(z, state, ref)
                if learn:
                    mean = mean + torch.randn(std.shape, device="cuda", generator=generator) * std
                action = grasp.constrained(mean[0], ref)
                acted += 1
        physical = frozen.post(action[None]).float().cpu().numpy()[0]
        requested = mapping.radians(physical)
        targets = np.clip(requested, env.low, env.high)
        if active and learn and not np.array_equal(targets, requested):
            actual = (targets - mapping.home) / mapping.scale + grasp.REFERENCE
            action = frozen.normalized(torch.as_tensor(actual, dtype=torch.float32, device="cuda"))
        for target in targets:
            _, _, ended, timeout, _ = env.step(env.action_for_targets(target))
            tracker.update(ended)
            if transfer is not None:
                transfer.update(env, tracker)
            if ended or timeout:
                break
        if active and (tracker.grasp is not None or env.steps - start["step"] >= grasp.PHASE_STEPS or ended or timeout):
            result = "stable_lift" if tracker.grasp is not None else "phase_timeout"
            transfer = Handoff(stats_at_boundary(env, tracker), result)
            terminal = (snapshot, action)
        elif active and learn:
            previous = (snapshot, action, 0.0, grasp.GAMMA ** grasp.CHUNK)
    assert previous is None
    if learn and terminal is not None:
        # The rest of the episode runs under the fixed VLA, so the last grasp action
        # receives the final placement outcome as its terminal return.
        replay.append(grasp.store_transition(*terminal, float(ended), 0.0, None))
    return {
        "seed": seed, **tracker.report(), "steps": env.steps, "success": bool(ended),
        "gate": start, "actor_chunks": acted, "handoff": vars(transfer) if transfer else None,
        "placement_reward_recorded": float(ended) if learn and terminal is not None else None,
        "seconds": round(perf_counter() - begin, 1),
    }, replay


def save_candidate(learner, index, provenance):
    path = HERE / f"candidate-{index:02d}.pt"
    staged = path.with_suffix(".tmp")
    torch.save({
        "schema_version": 1, "online_episodes": index, "config": vars(learner.config),
        "actor_state_dict": learner.actor.state_dict(), "critic_state_dict": learner.critic.state_dict(),
        "actor_target_state_dict": learner.actor_target.state_dict(),
        "critic_target_state_dict": learner.critic_target.state_dict(),
        "actor_optimizer_state_dict": learner.actor_optim.state_dict(),
        "critic_optimizer_state_dict": learner.critic_optim.state_dict(),
        "learner": learner.status(), "reward": "terminal_stable_placement",
        "stage1": str(grasp.STAGE1 / "stage1.pt"), "vla_provenance": provenance,
        "policy_batch_size": 1, "gate_radius_meters": grasp.GATE_RADIUS,
        "control_chunk_steps": grasp.CHUNK, "phase_steps": grasp.PHASE_STEPS,
        "action_delta_bound": grasp.ACTION_DELTA,
    }, staged)
    staged.replace(path)
    return str(path)


def retrieve_actor(path, config):
    data = torch.load(path, map_location="cpu", weights_only=False)
    actor = grasp.RLTChunkActor(config).cuda()
    actor.load_state_dict(data["actor_state_dict"], strict=True)
    return actor.requires_grad_(False).eval()


def summary(rows):
    lifts = [row for row in rows if row["stable_lift_step"] is not None]
    wins = [row for row in rows if row["success"]]
    transferred = [row for row in rows if row["handoff"] and row["handoff"]["result"] == "stable_lift"]
    return {
        "episodes": len(rows), "stable_lift": len(lifts), "stable_placement": len(wins),
        "held_above_box": sum(row["held_above_box_step"] is not None for row in rows),
        "stable_lift_handoff": len(transferred),
        "back_to_table_after_lift_handoff": sum(row["handoff"]["first_drop_step"] is not None for row in transferred),
        "median_handoff_cube_gain_mm": (
            float(np.median([row["handoff"]["snapshot"]["cube_height_gain_mm"] for row in transferred]))
            if transferred else None
        ),
    }


def contrasts(reference, candidate):
    assert [row["seed"] for row in reference] == [row["seed"] for row in candidate]
    results = {}
    for label, compare in (
        ("stable_lift", lambda row: row["stable_lift_step"] is not None),
        ("stable_placement", lambda row: row["success"]),
    ):
        before = [compare(row) for row in reference]
        after = [compare(row) for row in candidate]
        results[label] = {"baseline": sum(before), "actor": sum(after),
                          "actor_only": sum(b and not a for a, b in zip(before, after)),
                          "baseline_only": sum(a and not b for a, b in zip(before, after))}
    return results


def main():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    old = json.loads((PILOT / "report.json").read_text())
    assert old["complete"]
    diagnostic = [a["seed"] for a, b in zip(old["baseline"], old["eval"]) if a["success"] != b["success"]]
    declared = diagnostic + WARMUP + ONLINE + DEVELOPMENT + TEST
    assert len(declared) == len(set(declared))
    source_demo = json.loads((grasp.DATA / "manifest.json").read_text())
    assert not set(declared) & {row["seed"] for row in source_demo["episodes"]}
    env = grasp.SO101BlockBox(ROOT / "menagerie")
    mapping = grasp.JointMapping(json.loads((ROOT / "official_block_follower.json").read_text()))
    begin = perf_counter()
    learner = grasp.Learner()
    hash_value = hashlib.sha256((PILOT / "grasp-actor-critic.pt").read_bytes()).hexdigest()
    report = {
        "complete": False, "batch_size": 1, "chunk_steps": grasp.CHUNK,
        "reward": "terminal_stable_placement", "first_pilot_checkpoint_sha256": hash_value,
        "diagnostic_seeds": diagnostic, "warmup_seeds": WARMUP, "online_seeds": ONLINE,
        "development_seeds": DEVELOPMENT, "test_seeds": TEST,
        "diagnostic": {}, "warmup": [], "online": [], "candidates": {}, "development": {}, "test": {},
    }
    path = HERE / "report.json"

    def progress(stage, row, extra=None):
        grasp.save(path, report)
        line = {"stage": stage, "seed": row["seed"], "stable_lift": row["stable_lift_step"] is not None,
                "stable_placement": row["success"], "steps": row["steps"], "seconds": row["seconds"]}
        if extra:
            line.update(extra)
        print(json.dumps(line), flush=True)

    try:
        frozen = grasp.FrozenPolicy()
        old_actor = retrieve_actor(PILOT / "grasp-actor-critic.pt", learner.config)
        expected = {mode: {x["seed"]: x["success"] for x in old[mode]} for mode in ("baseline", "eval")}
        for label, actor in (("vla", None), ("v1", old_actor)):
            rows = report["diagnostic"][label] = []
            for seed in diagnostic:
                row, _ = rollout(env, mapping, frozen, seed, actor)
                expected_key = "baseline" if label == "vla" else "eval"
                row["matches_previous_outcome"] = row["success"] == expected[expected_key][seed]
                rows.append(row)
                progress("diagnostic_" + label, row)
        report["diagnostic_summary"] = {
            label: summary(rows) for label, rows in report["diagnostic"].items()
        }
        print(json.dumps({"diagnostic_summary": report["diagnostic_summary"]}), flush=True)
        del old_actor
        for index, seed in enumerate(WARMUP, 1):
            row, replay = rollout(env, mapping, frozen, seed, learn=True)
            learner.update(replay, train_actor=False)
            report["warmup"].append(row)
            progress("warmup", row, learner.status())
        assert learner.status()["positive_terminal_chunks"] >= 2, learner.status()
        for index, seed in enumerate(ONLINE, 1):
            row, replay = rollout(env, mapping, frozen, seed, learner.actor, learn=True)
            learner.update(replay)
            report["online"].append(row)
            if index in CHECKPOINT_STEPS:
                report["candidates"][str(index)] = save_candidate(learner, index, frozen.provenance)
            progress("online", row, {"index": index, **learner.status()})
        report["learner"] = learner.status()
        for label in ("vla", *report["candidates"]):
            actor = None if label == "vla" else retrieve_actor(report["candidates"][label], learner.config)
            rows = report["development"][label] = []
            for seed in DEVELOPMENT:
                row, _ = rollout(env, mapping, frozen, seed, actor)
                rows.append(row)
                progress("development_" + label, row)
        scores = {label: summary(rows) for label, rows in report["development"].items()}
        # Ties prefer the earlier checkpoint. A candidate must beat the VLA
        # on development placement count to be selected for possible deployment.
        selected = max(report["candidates"], key=lambda label: (scores[label]["stable_placement"], -int(label)))
        report["selection"] = {
            "criterion": "development stable placement count", "scores": scores,
            "actor_candidate": selected, "checkpoint": report["candidates"][selected],
            "beats_vla_on_development": scores[selected]["stable_placement"] > scores["vla"]["stable_placement"],
        }
        print(json.dumps({"selection": report["selection"]}), flush=True)
        grasp.save(path, report)
        actor = retrieve_actor(report["candidates"][selected], learner.config)
        for label, policy in (("vla", None), ("actor", actor)):
            rows = report["test"][label] = []
            for seed in TEST:
                row, _ = rollout(env, mapping, frozen, seed, policy)
                rows.append(row)
                progress("test_" + label, row)
        report["test_summary"] = {label: summary(rows) for label, rows in report["test"].items()}
        report["test_comparison"] = contrasts(report["test"]["vla"], report["test"]["actor"])
        report["complete"] = True
        report["seconds"] = round(perf_counter() - begin, 1)
        grasp.save(path, report)
        print(json.dumps({key: report[key] for key in (
            "diagnostic_summary", "selection", "test_summary", "test_comparison", "learner", "seconds"
        )}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
