"""Resume episode 640 after collecting 20 new SFT warmup episodes."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "rlt-stage2-new-encoder-640-resume-b1-20261124"))
import continue_fresh_actor_640 as continuation

fresh = continuation.fresh
grasp = fresh.grasp
SOURCE = continuation.SOURCE
START = continuation.START
INTERVAL = continuation.INTERVAL
INITIAL_CRITIC_WARMUP = fresh.CRITIC_WARMUP
EXTRA_CRITIC_UPDATES = 300
WARMUP_SEEDS = tuple(range(66000, 66020))
LATEST = HERE / "latest.pt"
REPORT = HERE / "report.json"
RATES = HERE / "success-rates.json"


def save(learner, report):
    with patch.multiple(continuation, HERE=HERE, LATEST=LATEST, REPORT=REPORT, RATES=RATES):
        continuation.save(learner, report)


def warmup(learner, report, frozen):
    rows = report["continuation_warmup"]
    if report["continuation_critic_warmup_updates"]:
        assert len(rows) == len(WARMUP_SEEDS)
        assert report["continuation_critic_warmup_updates"] == EXTRA_CRITIC_UPDATES
        return
    assert report["episodes_done"] == START
    expected_actor_updates = START * fresh.base.UPDATES_PER_EPISODE // 2
    expected_critic_updates = START * fresh.base.UPDATES_PER_EPISODE + INITIAL_CRITIC_WARMUP
    assert learner.actor_updates == expected_actor_updates
    assert learner.updates == expected_critic_updates
    if len(rows) < len(WARMUP_SEEDS):
        env = grasp.SO101BlockBox(ROOT / "menagerie")
        mapping_path = ROOT.parent / "config" / "official_block_follower.json"
        mapping = grasp.JointMapping(json.loads(mapping_path.read_text()))
        try:
            for seed in WARMUP_SEEDS[len(rows):]:
                row, replay = fresh.base.placement.rollout(env, mapping, frozen, seed, learn=True)
                learner.add(replay)
                rows.append(row)
                save(learner, report)
                print(json.dumps({"warmup_episode": len(rows), "warmup_target": len(WARMUP_SEEDS),
                                  "online_episode": START, "seed": seed, "success": row["success"],
                                  **learner.status()}), flush=True)
        finally:
            env.close()
    learner.fit(EXTRA_CRITIC_UPDATES, train_actor=False)
    assert learner.actor_updates == expected_actor_updates
    assert learner.updates == expected_critic_updates + EXTRA_CRITIC_UPDATES
    report["continuation_critic_warmup_updates"] = EXTRA_CRITIC_UPDATES
    save(learner, report)
    print(json.dumps({"warmup_complete": True, "warmup_episodes": len(rows),
                      "warmup_successes": sum(row["success"] for row in rows),
                      "continuation_critic_warmup_updates": EXTRA_CRITIC_UPDATES,
                      **learner.status()}), flush=True)


def restore(frozen, encoder_hash):
    path = LATEST if LATEST.exists() else SOURCE
    learner, report = fresh.job.restore_episode_checkpoint(path, frozen, encoder_hash)
    assert report["actor_critic_init"] == "fresh_reference_actor_random_critics"
    assert report["critic_warmup_updates"] == INITIAL_CRITIC_WARMUP
    if path == SOURCE:
        assert report["episodes_done"] == START
        assert len(report["warmup"]) == 12
        training_seeds = set(range(fresh.prior.ONLINE_BASE_SEED,
                                   fresh.prior.ONLINE_BASE_SEED + continuation.TARGET))
        assert not set(WARMUP_SEEDS) & (training_seeds | {row["seed"] for row in report["warmup"]})
        report["source_archives"] = report["archives"]
        report.update(archives={}, archive_every_episodes=INTERVAL,
                      continuation_start=START, continuation_source=str(SOURCE),
                      continuation_warmup=[], continuation_critic_warmup_updates=0)
    else:
        assert report["continuation_source"] == str(SOURCE)
    rows = report["continuation_warmup"]
    assert tuple(row["seed"] for row in rows) == WARMUP_SEEDS[:len(rows)]
    print(json.dumps({"phase": "resume_warmup", "warmup_done": len(rows),
                      "warmup_target": len(WARMUP_SEEDS), "online_episode": report["episodes_done"],
                      "encoder_step": frozen.encoder_step}), flush=True)
    warmup(learner, report, frozen)
    return learner, report


if __name__ == "__main__":
    with patch.multiple(fresh, open_state=restore, save=save, ARCHIVE_EVERY=INTERVAL,
                        CRITIC_WARMUP=INITIAL_CRITIC_WARMUP + EXTRA_CRITIC_UPDATES):
        fresh.main()
