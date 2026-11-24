"""Resume the fresh actor/critic run from its archived episode 640."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE.parent / "rlt-stage2-fresh-encoder20k-1024episodes-b1-20261123"
sys.path.insert(0, str(SOURCE_DIR))
import train_fresh_actor_1024 as fresh

START = 640
INTERVAL = 100
TARGET = fresh.TARGET_EPISODES
SOURCE = SOURCE_DIR / "episode-0640.pt"
LATEST = HERE / "latest.pt"
REPORT = HERE / "report.json"
RATES = HERE / "success-rates.json"


def restore(frozen, encoder_hash):
    path = LATEST if LATEST.exists() else SOURCE
    learner, report = fresh.job.restore_episode_checkpoint(path, frozen, encoder_hash)
    assert report["actor_critic_init"] == "fresh_reference_actor_random_critics"
    assert report["critic_warmup_updates"] == fresh.CRITIC_WARMUP
    if path == SOURCE:
        assert report["episodes_done"] == START
        report["source_archives"] = report["archives"]
        report.update(archives={}, archive_every_episodes=INTERVAL,
                      continuation_start=START, continuation_source=str(SOURCE))
    else:
        assert report["continuation_source"] == str(SOURCE)
    return learner, report


def save(learner, report):
    count = report["episodes_done"]
    assert count == len(report["online"]) and report["pending_updates"] == 0
    windows = []
    for end in range(START + INTERVAL, count + 1, INTERVAL):
        rows = report["online"][end - INTERVAL:end]
        successes = sum(bool(row["success"]) for row in rows)
        windows.append(dict(first_episode=end - INTERVAL + 1, last_episode=end,
                            episodes=INTERVAL, successes=successes,
                            success_rate=successes / INTERVAL))
    report["success_rate_windows"] = windows
    archive = HERE / f"episode-{count:04d}.pt"
    publish = count > START and ((count - START) % INTERVAL == 0 or count == TARGET)
    if publish:
        report["archives"][str(count)] = str(archive)
    report["learner"] = learner.status()
    with patch.object(fresh.prior, "archive_path", lambda _: LATEST):
        fresh.prior.save(count, learner, report)
    if publish and not archive.exists():
        os.link(LATEST, archive)
        if (count - START) % INTERVAL == 0:
            print(json.dumps({"success_rate_window": windows[-1]}), flush=True)
    fresh.grasp.save(REPORT, report)
    fresh.grasp.save(RATES, {"metric": "stable_placement", "windows": windows})


if __name__ == "__main__":
    with patch.multiple(fresh, open_state=restore, save=save, ARCHIVE_EVERY=INTERVAL):
        fresh.main()
