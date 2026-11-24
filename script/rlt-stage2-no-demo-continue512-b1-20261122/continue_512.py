"""Continue the frozen-encoder B1 actor-critic up to 800 episodes without evaluation."""
import json
from pathlib import Path
import sys
from time import perf_counter

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "rlt-stage2-no-demo-continue64-b1-20261122"))
import continue_no_demo as base

START = HERE / "start.pt"
LATEST = HERE / "latest.pt"
REPORT = HERE / "report.json"
SEEDS = range(57000, 57800)
SAVE_INTERVAL = 128


def record(report, **event):
    base.grasp.save(REPORT, report)
    print(json.dumps(event), flush=True)


def main():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    begin = perf_counter()
    anchor = torch.load(START, map_location="cpu", weights_only=False)
    origin = len(anchor["continuation_history"])
    origin_updates = dict(anchor["learner"])
    del anchor
    if REPORT.exists():
        report = json.loads(REPORT.read_text())
        assert report["starting_continuation_episodes"] == origin
        if report["complete"] and report["episodes_done"] == len(SEEDS):
            return
        report.update(complete=False, episodes_target=len(SEEDS))
        report.pop("elapsed_this_process_seconds", None)
    else:
        report = {
            "complete": False, "episodes_target": len(SEEDS), "episodes_done": 0,
            "starting_continuation_episodes": origin, "start_checkpoint": str(START),
            "start_learner": origin_updates, "archive_interval": SAVE_INTERVAL,
            "evaluation": False, "batch_size": 1, "archives": {},
        }
    env = base.grasp.SO101BlockBox(ROOT / "menagerie")
    mapping = base.grasp.JointMapping(json.loads((ROOT / "official_block_follower.json").read_text()))
    try:
        frozen = base.grasp.FrozenPolicy()
        resume = LATEST if LATEST.exists() else START
        learner, history = base.restore(resume, frozen.provenance["policy_version"])
        done = len(history) - origin
        assert 0 <= done <= len(SEEDS)
        assert [row["seed"] for row in history[origin:]] == list(SEEDS[:done])
        if done and (done % SAVE_INTERVAL == 0 or done == len(SEEDS)):
            archived = HERE / f"additional-{done:03d}.pt"
            if not archived.exists():
                base.persist(archived, learner, history, frozen)
            report["archives"][str(done)] = str(archived)
        report.update(episodes_done=done, learner=learner.status())
        record(report, stage="training", started=True, restored_episodes=origin, completed_in_this_run=done,
               target=len(SEEDS), archive_interval=SAVE_INTERVAL, **learner.status())
        for episode, seed in enumerate(SEEDS[done:], start=done + 1):
            row, replay = base.placement.rollout(env, mapping, frozen, seed, learner.actor, learn=True)
            learner.add(replay)
            learner.fit(base.UPDATES_PER_EPISODE, train_actor=True)
            history.append(row)
            # Keep one rolling atomic recovery file so interruptions lose at most one episode.
            base.persist(LATEST, learner, history, frozen)
            if episode % SAVE_INTERVAL == 0 or episode == len(SEEDS):
                archived = HERE / f"additional-{episode:03d}.pt"
                base.persist(archived, learner, history, frozen)
                report["archives"][str(episode)] = str(archived)
            current = history[origin:]
            report.update(episodes_done=episode, learner=learner.status(),
                          placements=sum(row["success"] for row in current),
                          stable_lifts=sum(row["stable_lift_step"] is not None for row in current))
            record(report, stage="training", additional_episode=episode, seed=seed, success=row["success"],
                   **learner.status())
        report["complete"] = True
        report["elapsed_this_process_seconds"] = round(perf_counter() - begin, 1)
        record(report, stage="complete", **report)
    finally:
        env.close()


if __name__ == "__main__":
    main()
