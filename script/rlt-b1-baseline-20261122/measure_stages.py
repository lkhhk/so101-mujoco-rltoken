"""Measure SO-101 grasp, transport and placement with frozen PI05 at B1."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from evaluate_so101_mujoco_lora import BASE_SHA, JointMapping, predict
from pi05_rl_runtime import load_ira_lora, make_processors
from so101_mujoco_blockbox import BOX_HALF, BOX_RIM, CAMERAS, CUBE_HALF, SO101BlockBox


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "scripted-mixed77"
BASE = ROOT.parent / "ira-pi05-v7-full-v2-008000-blockbox-20261117"
ADAPTER = ROOT / "grpo-000140-sft-mixed77-v1/checkpoints/010000"
TOKENIZER = ROOT.parent / "paligemma-tokenizer-35e4f464"
ADAPTER_SHA = "23707c30f61b5ba02f505c461ac54b4351739acdd47db35ac29f163faeece56a"
LIFT_METERS = 0.02
HOLD_STEPS = 6


class Stages:
    def __init__(self, env):
        self.env = env
        self.jaws = {env.model.body(name).id for name in ("gripper", "moving_jaw_so101_v1")}
        self.box = env.model.body("box").id
        self.initial_z = self.position()[2]
        self.hold_count = self.longest_hold = self.longest_bilateral = self.bilateral_count = 0
        self.first_contact = self.grasp = self.above_box = self.placed = None
        self.highest = float(self.initial_z)

    def position(self):
        return self.env.data.qpos[self.env.cube_pos:self.env.cube_pos + 3]

    def update(self, success):
        env = self.env
        position = self.position()
        self.highest = max(self.highest, float(position[2]))
        jaws = set()
        support_contact = False
        for contact in env.data.contact:
            geom1, geom2 = contact.geom
            if env.cube_geom not in (geom1, geom2):
                continue
            other = geom2 if geom1 == env.cube_geom else geom1
            body = env.model.geom_bodyid[other]
            if body in self.jaws:
                jaws.add(body)
            if body in (0, self.box):
                support_contact = True
        if jaws and self.first_contact is None:
            self.first_contact = env.steps
        lifted = position[2] - self.initial_z >= LIFT_METERS
        holding = bool(jaws) and lifted and not support_contact
        self.hold_count = self.hold_count + 1 if holding else 0
        self.bilateral_count = self.bilateral_count + 1 if holding and len(jaws) == 2 else 0
        self.longest_hold = max(self.longest_hold, self.hold_count)
        self.longest_bilateral = max(self.longest_bilateral, self.bilateral_count)
        if self.grasp is None and self.hold_count >= HOLD_STEPS:
            self.grasp = env.steps
        box_xy = env.data.mocap_pos[env.box_mocap, :2]
        if self.above_box is None and self.grasp is not None and holding:
            if np.all(np.abs(position[:2] - box_xy) <= BOX_HALF) and position[2] >= BOX_RIM + CUBE_HALF:
                self.above_box = env.steps
        if success:
            self.placed = env.steps

    def report(self):
        return {
            "first_jaw_contact_step": self.first_contact,
            "stable_lift_step": self.grasp,
            "held_above_box_step": self.above_box,
            "stable_placement_step": self.placed,
            "longest_lifted_contact_steps": self.longest_hold,
            "longest_bilateral_lifted_steps": self.longest_bilateral,
            "maximum_height_gain_mm": round((self.highest - self.initial_z) * 1000, 3),
        }


def mapped_step(env, mapping, command):
    requested = mapping.radians(command)
    clipped = np.clip(requested, env.low, env.high)
    _, _, success, time_limit, _ = env.step(env.action_for_targets(clipped))
    return success, time_limit, int(np.count_nonzero(np.abs(requested - clipped) > 1e-7))


def controls(env, mapping, manifest):
    row = manifest["episodes"][0]
    env.reset(seed=row["seed"])
    tracker = Stages(env)
    home = mapping.observation(env.data.qpos[env.position])
    for _ in range(30):
        success, _, _ = mapped_step(env, mapping, home)
        tracker.update(success)
    negative = tracker.report()
    assert negative["stable_lift_step"] is None and negative["stable_placement_step"] is None
    env.reset(seed=row["seed"])
    tracker = Stages(env)
    with np.load(DATA / row["directory"] / "motion.npz") as motion:
        for command in motion["action"]:
            success, _, _ = mapped_step(env, mapping, command)
            tracker.update(success)
            if success:
                break
    positive = tracker.report()
    assert all(positive[key] is not None for key in (
        "stable_lift_step", "held_above_box_step", "stable_placement_step"
    )), positive
    return {"scripted_positive_seed": row["seed"], "scripted_positive": positive, "home_hold_negative": negative}


def run_episode(env, mapping, policy, pre, post, seed, execute_steps):
    torch.manual_seed(seed)
    env.reset(seed=seed)
    tracker = Stages(env)
    stopped = False
    timings = []
    clamped = calls = 0
    begin = perf_counter()
    while not stopped:
        t0 = perf_counter()
        state = mapping.observation(env.data.qpos[env.position])
        images = [env.render_camera(name) for name in CAMERAS]
        chunk = predict(policy, pre, post, state, images)
        timings.append((perf_counter() - t0) * 1000)
        calls += 1
        for command in chunk[:execute_steps]:
            success, time_limit, clip_count = mapped_step(env, mapping, command)
            clamped += clip_count
            tracker.update(success)
            if success or time_limit:
                stopped = True
                break
    values = tracker.report()
    values.update({"seed": seed, "success": values["stable_placement_step"] is not None,
                   "steps": env.steps, "policy_calls": calls, "clipped_joint_targets": clamped,
                   "inference_p50_ms": round(float(np.median(timings)), 1),
                   "wall_seconds": round(perf_counter() - begin, 2)})
    return values


def summarize(episodes):
    grasp = [row for row in episodes if row["stable_lift_step"] is not None]
    success = [row for row in episodes if row["success"]]
    transport = [row for row in grasp if row["held_above_box_step"] is not None]
    return {
        "episodes": len(episodes), "successes": len(success),
        "jaw_contact": sum(row["first_jaw_contact_step"] is not None for row in episodes),
        "stable_lift": len(grasp), "held_above_box": len(transport),
        "successes_after_lift": sum(row["success"] for row in grasp),
        "successes_without_measured_lift": sum(row["stable_lift_step"] is None for row in success),
        "failure_before_lift": sum(row["stable_lift_step"] is None and not row["success"] for row in episodes),
        "failure_after_lift_before_transport": sum(
            row["held_above_box_step"] is None and not row["success"] for row in grasp
        ),
        "failure_after_transport": sum(not row["success"] for row in transport),
        "lift_sensitivity": {
            str(steps): sum(row["longest_lifted_contact_steps"] >= steps for row in episodes)
            for steps in (3, 6, 10)
        },
        "median_first_lift_seconds": None if not grasp else float(np.median([row["stable_lift_step"] for row in grasp]) / 30),
        "median_success_seconds": None if not success else float(np.median([row["steps"] for row in success]) / 30),
        "median_per_episode_inference_p50_ms": float(np.median([row["inference_p50_ms"] for row in episodes])),
    }


def save(path, report):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n")
    pending.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-steps", type=int, default=10, choices=(10, 50))
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(1)
    seeds = list(range(31000, 31000 + args.count))
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert not (set(seeds) & {row["seed"] for row in manifest["episodes"]})
    env = SO101BlockBox(ROOT / "menagerie")
    mapping = JointMapping(json.loads((ROOT / "official_block_follower.json").read_text()))
    output = HERE / f"c{args.execute_steps}-n{args.count}.json"
    try:
        spec = {"batch_size": 1, "execute_steps": args.execute_steps,
                "seeds": seeds, "control_hz": 30, "max_steps": env.max_steps,
                "lift_meters": LIFT_METERS, "hold_steps": HOLD_STEPS}
        prior = json.loads(output.read_text()) if output.exists() else None
        if prior:
            assert prior["spec"] == spec
            if prior["complete"]:
                print(json.dumps(prior["summary"]), flush=True)
                return
        diagnostics = controls(env, mapping, manifest)
        print(json.dumps({"stage_tracker_controls": diagnostics}), flush=True)
        policy, provenance = load_ira_lora(BASE, ADAPTER, device="cuda")
        assert provenance["base_sha256"] == BASE_SHA and provenance["adapter_sha256"] == ADAPTER_SHA
        policy.requires_grad_(False).eval()
        pre, post = make_processors(policy, ADAPTER, TOKENIZER, "cuda")
        episodes = prior["episodes"] if prior else []
        assert [row["seed"] for row in episodes] == seeds[:len(episodes)]
        for seed in seeds[len(episodes):]:
            row = run_episode(env, mapping, policy, pre, post, seed, args.execute_steps)
            episodes.append(row)
            summary = summarize(episodes)
            save(output, {"spec": spec, "provenance": provenance, "controls": diagnostics,
                          "episodes": episodes, "summary": summary, "complete": len(episodes) == len(seeds)})
            print(json.dumps({"episode": len(episodes), **row}), flush=True)
        print(json.dumps({"summary": summary}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
