"""Convert successful scripted episodes to the Stage 2 B1 replay schema."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RLT_ROOT = HERE.parent
sys.path.insert(0, str(RLT_ROOT / "rlt-stage2-transfer-b1-20261122"))
import train_placement_credit as placement

grasp = placement.grasp
DEMO_COUNT = 24
REPEAT_SEEDS = (53000, 53001, 53002)
FIELDS = ("z", "state", "reference", "executed", "reward", "discount", "next_z", "next_state", "next_reference")


@torch.no_grad()
def convert_episode(env, mapping, frozen, row):
    seed = row["seed"]
    torch.manual_seed(seed)
    env.reset(seed=seed)
    tracker = grasp.Stages(env)
    gate = handoff = pending = terminal = None
    replay = []
    error = []
    start = perf_counter()
    with np.load(grasp.DATA / row["directory"] / "motion.npz") as motion:
        commands = motion["action"]
    assert commands.shape == (row["frames"], 6) and np.isfinite(commands).all()
    complete = False
    for offset in range(0, len(commands), grasp.CHUNK):
        batch = commands[offset:offset + grasp.CHUNK]
        if gate is None and tracker.grasp is None and grasp.gate(env, tracker):
            gate = env.steps
        active = gate is not None and handoff is None
        if active:
            assert len(batch) == grasp.CHUNK
            ref, (token, state) = frozen.observe(env, mapping, with_token=True)
            snapshot = (token, state, ref)
            if pending is not None:
                replay.append(grasp.store_transition(*pending, snapshot))
                pending = None
        executed_targets = []
        for command in batch:
            target = np.clip(mapping.radians(command), env.low, env.high)
            executed_targets.append(target)
            _, _, complete, _, _ = env.step(env.action_for_targets(target))
            tracker.update(complete)
            if complete:
                break
        if active:
            targets = np.asarray(executed_targets)
            assert targets.shape == (grasp.CHUNK, 6)
            actual = (targets - mapping.home) / mapping.scale + grasp.REFERENCE
            executed = frozen.normalized(torch.as_tensor(actual, dtype=torch.float32, device="cuda"))
            error.append((executed - ref).abs().cpu().numpy())
            if tracker.grasp is not None:
                handoff = env.steps
                assert handoff - gate <= grasp.PHASE_STEPS
                terminal = (snapshot, executed)
            else:
                pending = (snapshot, executed, 0.0, grasp.GAMMA ** grasp.CHUNK)
        if complete:
            break
    assert complete and env.steps == row["frames"], (seed, env.steps, row["frames"])
    assert gate is not None and handoff is not None and gate < tracker.first_contact < tracker.grasp < tracker.placed
    assert pending is None and terminal is not None
    replay.append(grasp.store_transition(*terminal, 1.0, 0.0, None))
    delta = np.asarray(error)
    assert delta.shape == (len(replay), grasp.CHUNK, 6) and np.isfinite(delta).all()
    within = (delta <= grasp.ACTION_DELTA + 1e-6)
    metrics = {
        "episode": row["episode"], "seed": seed, "gate_step": gate, "handoff_step": handoff,
        **tracker.report(), "chunks": len(replay), "reference_action_component_mae": float(delta.mean()),
        "elements_outside_actor_bound": int((~within).sum()),
        "elements_total": int(delta.size),
        "fully_reachable_chunks": int(within.all(axis=(1, 2)).sum()),
        "terminal_chunk_inside_bound": bool(within[-1].all()),
        "wall_seconds": round(perf_counter() - start, 2),
    }
    return replay, metrics


def replay_digest(rows):
    digest = sha256()
    for row in rows:
        for item in row:
            digest.update(item.contiguous().numpy().tobytes())
    return digest.hexdigest()


def review(replay, expect_terminal):
    assert replay
    expected_shapes = ((2048,), (6,), (10, 6), (10, 6), (1,), (1,), (2048,), (6,), (10, 6))
    for row in replay:
        assert len(row) == len(FIELDS) and tuple(tuple(x.shape) for x in row) == expected_shapes
        assert all(torch.isfinite(x).all() for x in row)
    if expect_terminal:
        assert sum(row[4].item() for row in replay) == 1
        assert replay[-1][4].item() == 1 and replay[-1][5].item() == 0
        assert sum(row[5].item() == 0 for row in replay) == 1


def save_replay(path, entries, meta):
    staged = path.with_suffix(".tmp")
    torch.save({"schema_version": 1, "fields": FIELDS, "episodes": entries, **meta}, staged)
    staged.replace(path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["fields"] == FIELDS and len(loaded["episodes"]) == len(entries)
    for saved, source in zip(loaded["episodes"], entries):
        review(saved["replay"], expect_terminal=meta["source"] == "scripted")
        assert replay_digest(saved["replay"]) == replay_digest(source["replay"])


def repeatability(env, mapping, frozen):
    entries, pairs = [], []
    fields = ("stable_lift_step", "stable_placement_step", "steps")
    for seed in REPEAT_SEEDS:
        runs = []
        for attempt in (1, 2):
            row, replay = placement.rollout(env, mapping, frozen, seed, learn=True)
            if replay:
                review(replay, expect_terminal=row["success"])
            digest = replay_digest(replay)
            runs.append({
                "attempt": attempt, **{key: row[key] for key in fields},
                "success": row["success"], "chunks": len(replay), "replay_sha256": digest,
            })
            if attempt == 1 and replay:
                entries.append({"seed": seed, "episode": None, "replay": replay})
            print(json.dumps({"reference_repeat": seed, **runs[-1]}), flush=True)
        pairs.append({
            "seed": seed, "runs": runs,
            "outcome_equal": runs[0]["success"] == runs[1]["success"],
            "events_equal": all(runs[0][field] == runs[1][field] for field in fields),
            "replay_exact": runs[0]["replay_sha256"] == runs[1]["replay_sha256"],
        })
    return pairs, entries


def main():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    manifest_bytes = (grasp.DATA / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    by_id = {row["episode"]: row for row in manifest["episodes"]}
    # Use a fixed sample of training episodes only, with original and supplemental demos.
    rng = np.random.default_rng(grasp.SEED)
    chosen = sorted(rng.choice(manifest["train"], DEMO_COUNT, replace=False).tolist())
    assert len(chosen) == DEMO_COUNT and not set(chosen) & set(manifest["validation"])
    assert not set(REPEAT_SEEDS) & {row["seed"] for row in manifest["episodes"]}
    env = grasp.SO101BlockBox(RLT_ROOT / "menagerie")
    mapping = grasp.JointMapping(json.loads((RLT_ROOT / "official_block_follower.json").read_text()))
    try:
        frozen = grasp.FrozenPolicy()
        demo, reports = [], []
        provenance = {
            "batch_size": 1, "chunk_steps": grasp.CHUNK, "gate_radius_m": grasp.GATE_RADIUS,
            "action_delta_bound": grasp.ACTION_DELTA, "phase_steps": grasp.PHASE_STEPS,
            "reward": "final_stable_placement", "stage1_step": frozen.encoder_step,
            "vla_provenance": frozen.provenance, "demo_manifest_sha256": sha256(manifest_bytes).hexdigest(),
        }
        for episode in chosen:
            row = by_id[episode]
            replay, metrics = convert_episode(env, mapping, frozen, row)
            review(replay, expect_terminal=True)
            demo.append({"episode": episode, "seed": row["seed"], "replay": replay})
            reports.append(metrics)
            print(json.dumps({"converted": len(demo), **metrics}), flush=True)
        save_replay(HERE / "scripted-replay.pt", demo, {**provenance, "source": "scripted"})
        grasp.save(HERE / "report.json", {
            "complete": False, "source": provenance, "episodes": reports,
            "scripted_replay": str(HERE / "scripted-replay.pt"),
        })
        repeats, baseline = repeatability(env, mapping, frozen)
        if baseline:
            save_replay(HERE / "reference-repeat-replay.pt", baseline, {**provenance, "source": "reference"})
        elements = sum(row["elements_total"] for row in reports)
        total_chunks = sum(row["chunks"] for row in reports)
        metrics = {
            "demo_episodes": len(reports), "demo_chunks": total_chunks, "positive_terminal_chunks": len(reports),
            "fully_reachable_chunks": sum(row["fully_reachable_chunks"] for row in reports),
            "elements_outside_actor_bound_fraction": sum(row["elements_outside_actor_bound"] for row in reports) / elements,
            "terminal_chunks_inside_actor_bound": sum(row["terminal_chunk_inside_bound"] for row in reports),
            "reference_repeat_outcomes_equal": sum(row["outcome_equal"] for row in repeats),
            "reference_repeat_events_equal": sum(row["events_equal"] for row in repeats),
            "reference_repeat_exact_chunks": sum(row["replay_exact"] for row in repeats),
        }
        grasp.save(HERE / "report.json", {
            "complete": True, "source": provenance, "episodes": reports, "summary": metrics,
            "reference_repeats": repeats, "scripted_replay": str(HERE / "scripted-replay.pt"),
            "reference_repeat_replay": str(HERE / "reference-repeat-replay.pt") if baseline else None,
        })
        print(json.dumps({"summary": metrics}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
