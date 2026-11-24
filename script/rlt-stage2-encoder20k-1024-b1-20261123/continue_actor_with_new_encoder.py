"""Continue existing B1 actor-critic with the frozen +20,000-step RL Token encoder."""
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = ROOT / "rlt-stage2-no-demo-continue512-b1-20261122"
sys.path.insert(0, str(RUN))
import continue_512 as previous
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTokenEncoder

base = previous.base
grasp = base.grasp
START = RUN / "additional-800.pt"
ENCODER = ROOT / "rlt-stage1-30000-b64-20261123" / "additional-20000-total-20512.pt"
TARGET = 1024
SAVE_EVERY = 128
WARMUP = list(range(60000, 60012))
ONLINE_BASE_SEED = 60020
REPORT = HERE / "report.json"


def file_sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_encoder():
    data = torch.load(ENCODER, map_location="cpu", mmap=True, weights_only=False)
    assert data["rlt_architecture"] == "paper_v1" and data["image_only"]
    assert data["step"] == 20512 and data["additional_steps"] == 20000
    config = grasp.PI05RLTConfig(mode="inference")
    for name in ("rlt_architecture", "vlm_hidden_dim", "rlt_hidden_dim", "rlt_encoder_layers", "rlt_num_heads"):
        assert data["config"][name] == getattr(config, name)
    encoder = RLTokenEncoder(config).cuda()
    encoder.load_state_dict(data["encoder_state_dict"], strict=True)
    return encoder.requires_grad_(False).eval(), data["step"]


def archive_path(index):
    return HERE / (f"additional-{index:04d}.pt" if index else "encoder-warmup.pt")


def report_status(r, learner, stage):
    r["stage"] = stage
    r["learner"] = learner.status()
    grasp.save(REPORT, r)
    print(json.dumps({"stage": stage, "warmup_episodes": len(r["warmup"]),
                      "online_episodes": len(r["online"]), "additional_updates": r["additional_updates"],
                      "pending_updates": r["pending_updates"], **r["learner"]}), flush=True)


def save(index, learner, report):
    path = archive_path(index)
    base.pilot.atomic_torch(path, {
        "schema_version": 1, "arm": "vla_warmup", "policy_batch_size": 1,
        "chunk_steps": grasp.CHUNK, "action_delta_bound": grasp.ACTION_DELTA,
        "rlt_checkpoint": str(ENCODER), "rl_token_checkpoint": str(ENCODER),
        "actor_state_dict": learner.actor.state_dict(), "critic_state_dict": learner.critic.state_dict(),
        "actor_target_state_dict": learner.actor_target.state_dict(),
        "critic_target_state_dict": learner.critic_target.state_dict(),
        "actor_optimizer_state_dict": learner.actor_optim.state_dict(),
        "critic_optimizer_state_dict": learner.critic_optim.state_dict(), "config": vars(learner.config),
        "learner": learner.status(), "replay": learner.replay, "is_demo": learner.is_demo,
        "rng": {"sampler": learner.prng.getstate(), "noise": learner.noise.get_state()},
        "report": report, "reward": "stable_placement_after_frozen_vla_handoff",
        "vla_provenance": report["vla_provenance"],
    })
    return str(path)


def restore(frozen, encoder_hash):
    candidates = sorted(HERE.glob("additional-[0-9][0-9][0-9][0-9].pt"), reverse=True)
    if archive_path(0).exists():
        candidates.append(archive_path(0))
    if not candidates:
        learner, _ = base.restore(START, frozen.provenance["policy_version"])
        original_chunks = len(learner.replay)
        learner.replay.clear()
        learner.is_demo.clear()
        assert (learner.updates, learner.actor_updates) == (69340, 34520)
        return learner, {
            "complete": False, "additional_updates": 0, "pending_updates": 0,
            "target": TARGET, "save_every": SAVE_EVERY, "batch_size": 1,
            "base_actor_critic": str(START), "encoder": str(ENCODER), "encoder_sha256": encoder_hash,
            "encoder_total_step": frozen.encoder_step, "discarded_old_encoder_replay_chunks": original_chunks,
            "vla_provenance": frozen.provenance,
            "warmup": [], "online": [], "archives": {},
        }
    saved = torch.load(candidates[0], map_location="cpu", weights_only=False)
    report = saved["report"]
    assert report["encoder_sha256"] == encoder_hash and report["target"] == TARGET
    assert report["vla_provenance"]["policy_version"] == frozen.provenance["policy_version"]
    assert (saved["learner"]["critic_updates"], saved["learner"]["actor_updates"]) == (
        69340 + report["additional_updates"], 34520 + report["additional_updates"] // 2
    )
    learner = base.pilot.TrialLearner()
    for name in ("actor", "critic", "actor_target", "critic_target"):
        getattr(learner, name).load_state_dict(saved[name + "_state_dict"], strict=True)
    for name in ("actor", "critic"):
        getattr(learner, name + "_optim").load_state_dict(saved[name + "_optimizer_state_dict"])
    assert not any(saved["is_demo"])
    learner.add(saved["replay"])
    learner.prng.setstate(saved["rng"]["sampler"])
    learner.noise.set_state(saved["rng"]["noise"])
    learner.updates = saved["learner"]["critic_updates"]
    learner.actor_updates = saved["learner"]["actor_updates"]
    learner.latest = {key: saved["learner"][key] for key in ("critic_loss", "actor_q", "anchor_mse")
                      if key in saved["learner"]}
    return learner, report


def main():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    encoder_hash = file_sha256(ENCODER)
    state = torch.load(ENCODER, map_location="cpu", mmap=True, weights_only=False)
    assert state["step"] == 20512
    encoder_policy_version = state["cache_provenance"]["policy_version"]
    del state
    with patch.object(grasp, "load_encoder", load_encoder):
        frozen = grasp.FrozenPolicy()
    assert encoder_policy_version == frozen.provenance["policy_version"]
    learner, report = restore(frozen, encoder_hash)
    if report["complete"]:
        return
    env = grasp.SO101BlockBox(ROOT / "menagerie")
    mapping = grasp.JointMapping(json.loads((ROOT / "official_block_follower.json").read_text()))
    begin = perf_counter()
    try:
        report_status(report, learner, "ready")
        if not archive_path(0).exists():
            for seed in WARMUP:
                row, replay = base.placement.rollout(env, mapping, frozen, seed, learn=True)
                learner.add(replay)
                report["warmup"].append(row)
                report_status(report, learner, "collect_warmup")
            if len(learner.replay) < 32 or learner.status()["positive_terminal_chunks"] == 0:
                raise RuntimeError("No usable successful warmup data for new encoder")
            report["warmup_recovery"] = str(archive_path(0))
            save(0, learner, report)
        while report["additional_updates"] < TARGET:
            if report["pending_updates"] == 0:
                seed = ONLINE_BASE_SEED + len(report["online"])
                row, replay = base.placement.rollout(env, mapping, frozen, seed, learner.actor, learn=True)
                learner.add(replay)
                report["online"].append(row)
                report["pending_updates"] = min(base.UPDATES_PER_EPISODE, TARGET - report["additional_updates"])
                report_status(report, learner, "online_rollout")
            next_save = (report["additional_updates"] // SAVE_EVERY + 1) * SAVE_EVERY
            steps = min(report["pending_updates"], next_save - report["additional_updates"])
            learner.fit(steps, train_actor=True)
            report["additional_updates"] += steps
            report["pending_updates"] -= steps
            index = report["additional_updates"]
            assert (learner.updates, learner.actor_updates) == (69340 + index, 34520 + index // 2)
            if index % SAVE_EVERY == 0:
                report["archives"][str(index)] = str(archive_path(index))
                report["complete"] = index == TARGET
                report["elapsed_this_process_seconds"] = round(perf_counter() - begin, 1)
                save(index, learner, report)
                report_status(report, learner, "complete" if report["complete"] else "checkpoint")
            else:
                report_status(report, learner, "training")
    finally:
        env.close()


if __name__ == "__main__":
    main()
