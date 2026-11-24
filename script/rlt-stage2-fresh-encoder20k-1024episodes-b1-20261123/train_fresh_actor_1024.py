"""Train new actor and critics from the frozen SFT policy and +20k RL Token encoder."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREVIOUS = ROOT / "rlt-stage2-encoder20k-1024episodes-b1-20261123"
sys.path.insert(0, str(PREVIOUS))
import continue_actor_1024_episodes as job

prior, base, grasp = job.prior, job.base, job.grasp
TARGET_EPISODES = 1024
ARCHIVE_EVERY = 128
CRITIC_WARMUP = 300
REPORT = HERE / "report.json"
LATEST = HERE / "latest.pt"
WARMUP = prior.HERE / "encoder-warmup.pt"


def save(learner, report):
    report["learner"] = learner.status()
    # Reuse the established atomic recovery and hard-link archive writer in this run directory.
    with patch.multiple(job, HERE=HERE, REPORT=REPORT, LATEST=LATEST):
        job.save(learner, report)


def open_state(frozen, encoder_hash):
    if LATEST.exists():
        learner, report = job.restore_episode_checkpoint(LATEST, frozen, encoder_hash)
        assert report["actor_critic_init"] == "fresh_reference_actor_random_critics"
        assert report["critic_warmup_updates"] == CRITIC_WARMUP
        return learner, report
    saved = torch.load(WARMUP, map_location="cpu", mmap=True, weights_only=False)
    source_report = saved["report"]
    assert source_report["encoder_sha256"] == encoder_hash
    assert source_report["vla_provenance"]["policy_version"] == frozen.provenance["policy_version"]
    assert source_report["additional_updates"] == 0 and not source_report["online"]
    assert [row["seed"] for row in source_report["warmup"]] == prior.WARMUP
    assert len(saved["replay"]) >= 32 and not any(saved["is_demo"])
    learner = base.pilot.TrialLearner()
    assert learner.updates == learner.actor_updates == 0
    learner.add(saved["replay"])
    assert learner.status()["positive_terminal_chunks"] > 0
    learner.fit(CRITIC_WARMUP, train_actor=False)
    assert (learner.updates, learner.actor_updates) == (CRITIC_WARMUP, 0)
    return learner, {
        "complete": False, "episode_target": TARGET_EPISODES,
        "archive_every_episodes": ARCHIVE_EVERY, "batch_size": 1,
        "episodes_done": 0, "additional_updates": 0, "pending_updates": 0,
        "placements": 0, "stable_lifts": 0, "archives": {}, "online": [],
        "actor_critic_init": "fresh_reference_actor_random_critics",
        "encoder": str(prior.ENCODER), "encoder_sha256": encoder_hash,
        "encoder_total_step": frozen.encoder_step, "vla_provenance": frozen.provenance,
        "warmup_source": str(WARMUP), "warmup": source_report["warmup"],
        "critic_warmup_updates": CRITIC_WARMUP,
    }


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
    assert (learner.updates, learner.actor_updates) == (
        CRITIC_WARMUP + report["episodes_done"] * base.UPDATES_PER_EPISODE,
        report["episodes_done"] * base.UPDATES_PER_EPISODE // 2,
    )
    assert [row["seed"] for row in report["online"]] == list(
        range(prior.ONLINE_BASE_SEED, prior.ONLINE_BASE_SEED + report["episodes_done"])
    )
    if report["complete"]:
        return
    save(learner, report)
    print(json.dumps({"started": True, "episodes_done": report["episodes_done"],
                      "target": TARGET_EPISODES, "save_every_episodes": ARCHIVE_EVERY,
                      "initialization": report["actor_critic_init"],
                      "encoder_step": frozen.encoder_step, "warmup_episodes": len(report["warmup"]),
                      **learner.status()}), flush=True)
    env = grasp.SO101BlockBox(ROOT / "menagerie")
    mapping_path = ROOT.parent / "config" / "official_block_follower.json"
    mapping = grasp.JointMapping(json.loads(mapping_path.read_text()))
    try:
        while len(report["online"]) < TARGET_EPISODES:
            seed = prior.ONLINE_BASE_SEED + len(report["online"])
            row, replay = base.placement.rollout(env, mapping, frozen, seed, learner.actor, learn=True)
            learner.add(replay)
            learner.fit(base.UPDATES_PER_EPISODE, train_actor=True)
            report["online"].append(row)
            count = len(report["online"])
            assert (learner.updates, learner.actor_updates) == (
                CRITIC_WARMUP + count * base.UPDATES_PER_EPISODE, count * base.UPDATES_PER_EPISODE // 2)
            report.update(episodes_done=count, additional_updates=count * base.UPDATES_PER_EPISODE,
                          complete=count == TARGET_EPISODES, placements=report["placements"] + int(row["success"]),
                          stable_lifts=report["stable_lifts"] + int(row["stable_lift_step"] is not None))
            save(learner, report)
            print(json.dumps({"episode": count, "seed": seed, "success": row["success"],
                              "stable_lift": row["stable_lift_step"] is not None,
                              "target": TARGET_EPISODES, "archive": report["archives"].get(str(count)),
                              **learner.status()}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
