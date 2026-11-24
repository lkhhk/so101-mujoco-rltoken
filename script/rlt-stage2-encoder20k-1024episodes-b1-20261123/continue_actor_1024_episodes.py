"""Train the +20k-encoder B1 actor-critic for 1024 online episodes."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "rlt-stage2-encoder20k-1024-b1-20261123"
sys.path.insert(0, str(SOURCE))
import continue_actor_with_new_encoder as prior

base = prior.base
grasp = prior.grasp
TARGET_EPISODES = 1024
ARCHIVE_EVERY = 128
REPORT = HERE / "report.json"
LATEST = HERE / "latest.pt"


def restore_episode_checkpoint(path, frozen, encoder_hash):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    report = saved["report"]
    assert report["encoder_sha256"] == encoder_hash and report["episode_target"] == TARGET_EPISODES
    assert report["vla_provenance"]["policy_version"] == frozen.provenance["policy_version"]
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
    return learner, report


def open_state(frozen, encoder_hash):
    if LATEST.exists():
        return restore_episode_checkpoint(LATEST, frozen, encoder_hash)
    learner, report = prior.restore(frozen, encoder_hash)
    assert len(report["warmup"]) == len(prior.WARMUP) and len(learner.replay) >= 32
    report["source_step_archives"] = report.pop("archives")
    report.update(episode_target=TARGET_EPISODES, archive_every_episodes=ARCHIVE_EVERY,
                  complete=False, archives={})
    report["pending_updates"] = len(report["online"]) * base.UPDATES_PER_EPISODE - report["additional_updates"]
    assert 0 <= report["pending_updates"] <= base.UPDATES_PER_EPISODE
    report["source_online_episodes"] = len(report["online"])
    report["source_additional_updates"] = report["additional_updates"]
    return learner, report


def save(learner, report):
    count = report["episodes_done"]
    assert report["pending_updates"] == 0
    archive = HERE / f"episode-{count:04d}.pt"
    publish = count > 0 and count % ARCHIVE_EVERY == 0
    if publish:
        report["archives"][str(count)] = str(archive)
    report["learner"] = learner.status()
    with patch.object(prior, "archive_path", lambda _: LATEST):
        prior.save(count, learner, report)
    if publish and not archive.exists():
        os.link(LATEST, archive)
    grasp.save(REPORT, report)


def finish_pending(learner, report):
    count = report["pending_updates"]
    if count:
        learner.fit(count, train_actor=True)
        report["additional_updates"] += count
        report["pending_updates"] = 0
    report["episodes_done"] = len(report["online"])
    assert learner.updates == 69340 + report["episodes_done"] * base.UPDATES_PER_EPISODE
    assert learner.actor_updates == 34520 + report["episodes_done"] * base.UPDATES_PER_EPISODE // 2


def main():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    encoder_hash = prior.file_sha256(prior.ENCODER)
    metadata = torch.load(prior.ENCODER, map_location="cpu", mmap=True, weights_only=False)
    encoder_vla = metadata["cache_provenance"]["policy_version"]
    del metadata
    with patch.object(grasp, "load_encoder", prior.load_encoder):
        frozen = grasp.FrozenPolicy()
    assert frozen.provenance["policy_version"] == encoder_vla
    learner, report = open_state(frozen, encoder_hash)
    finish_pending(learner, report)
    save(learner, report)
    print(json.dumps({"started": True, "episodes_done": report["episodes_done"],
                      "target": TARGET_EPISODES, "save_every_episodes": ARCHIVE_EVERY,
                      "source_online_episodes": report["source_online_episodes"],
                      "source_additional_updates": report["source_additional_updates"],
                      "warmup_episodes": len(report["warmup"]), **learner.status()}), flush=True)
    if report["complete"]:
        return
    env = grasp.SO101BlockBox(ROOT / "menagerie")
    mapping = grasp.JointMapping(json.loads((ROOT / "official_block_follower.json").read_text()))
    try:
        while len(report["online"]) < TARGET_EPISODES:
            seed = prior.ONLINE_BASE_SEED + len(report["online"])
            row, replay = base.placement.rollout(env, mapping, frozen, seed, learner.actor, learn=True)
            learner.add(replay)
            report["online"].append(row)
            report["pending_updates"] = base.UPDATES_PER_EPISODE
            finish_pending(learner, report)
            count = report["episodes_done"]
            report["complete"] = count == TARGET_EPISODES
            report["placements"] = sum(x["success"] for x in report["online"])
            report["stable_lifts"] = sum(x["stable_lift_step"] is not None for x in report["online"])
            save(learner, report)
            print(json.dumps({"episode": count, "seed": seed, "success": row["success"],
                              "stable_lift": row["stable_lift_step"] is not None,
                              "target": TARGET_EPISODES, "archive": report["archives"].get(str(count)),
                              **learner.status()}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
