"""Resume the no-demonstration B1 actor-critic for up to 64 new episodes."""
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PILOT = ROOT / "rlt-stage2-demo-ab-b1-20261122"
sys.path.insert(0, str(PILOT))
import train_demo_ab as pilot

placement = pilot.placement
grasp = pilot.grasp
START = PILOT / "vla_warmup-episode-08.pt"
START_SHA = "37ee224542c553b441e0d612f1358aa980575c9a31dd3513e44404ba29fc1764"
REPORT = HERE / "report.json"
LATEST = HERE / "latest.pt"
ONLINE = list(range(56000, 56064))
DEVELOPMENT = list(range(56100, 56108))
TEST = list(range(56200, 56232))
TEST_REPEATS = 2
SAVE_AT = (16, 32, 64)
UPDATES_PER_EPISODE = 80


def path_for(index):
    return HERE / f"additional-{index:03d}.pt" if index else START


def restore(path, config):
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["arm"] == "vla_warmup"
    assert loaded["policy_batch_size"] == 1 and loaded["chunk_steps"] == grasp.CHUNK
    assert loaded["action_delta_bound"] == grasp.ACTION_DELTA
    assert loaded["reward"] == "stable_placement_after_frozen_vla_handoff"
    assert not any(loaded["is_demo"]) and len(loaded["is_demo"]) == len(loaded["replay"])
    assert loaded["vla_provenance"]["policy_version"] == config
    learner = pilot.TrialLearner()
    for name in ("actor", "critic", "actor_target", "critic_target"):
        getattr(learner, name).load_state_dict(loaded[name + "_state_dict"], strict=True)
    for name in ("actor", "critic"):
        getattr(learner, name + "_optim").load_state_dict(loaded[name + "_optimizer_state_dict"])
    learner.add(loaded["replay"])
    learner.prng.setstate(loaded["rng"]["sampler"])
    learner.noise.set_state(loaded["rng"]["noise"])
    learner.updates = loaded["learner"]["critic_updates"]
    learner.actor_updates = loaded["learner"]["actor_updates"]
    episode_history = loaded.get("continuation_history", [])
    assert learner.updates == 940 + len(episode_history) * UPDATES_PER_EPISODE
    assert learner.actor_updates == 320 + len(episode_history) * (UPDATES_PER_EPISODE // 2)
    return learner, episode_history


def persist(path, learner, history, frozen):
    pilot.atomic_torch(path, {
        "schema_version": 1, "arm": "vla_warmup", "online_episodes": 8 + len(history),
        "config": vars(learner.config), "actor_state_dict": learner.actor.state_dict(),
        "critic_state_dict": learner.critic.state_dict(), "actor_target_state_dict": learner.actor_target.state_dict(),
        "critic_target_state_dict": learner.critic_target.state_dict(),
        "actor_optimizer_state_dict": learner.actor_optim.state_dict(),
        "critic_optimizer_state_dict": learner.critic_optim.state_dict(),
        "learner": learner.status(), "replay": learner.replay, "is_demo": learner.is_demo,
        "rng": {"sampler": learner.prng.getstate(), "noise": learner.noise.get_state()},
        "reward": "stable_placement_after_frozen_vla_handoff", "vla_provenance": frozen.provenance,
        "rl_token_checkpoint": str(grasp.STAGE1 / "stage1.pt"), "policy_batch_size": 1,
        "chunk_steps": grasp.CHUNK, "action_delta_bound": grasp.ACTION_DELTA,
        "continuation_history": history, "start_checkpoint_sha256": START_SHA,
    })


def state_report():
    settings = {"online_seeds": ONLINE, "development_seeds": DEVELOPMENT, "test_seeds": TEST,
                "test_repeats": TEST_REPEATS, "warm_start_checkpoint": str(START),
                "updates_per_episode": UPDATES_PER_EPISODE, "save_at": SAVE_AT}
    # JSON round trips tuples as lists.
    settings["save_at"] = list(settings["save_at"])
    if REPORT.exists():
        value = json.loads(REPORT.read_text())
        assert value["settings"] == settings
        return value
    return {"complete": False, "settings": settings, "train": [], "development": {}, "test": {}}


def progress(report, stage, **detail):
    report["phase"] = stage
    grasp.save(REPORT, report)
    print(json.dumps({"stage": stage, **detail}), flush=True)


def candidate(label, learner):
    return None if label == "vla" else placement.retrieve_actor(path_for(int(label)), learner.config)


def evaluate(env, mapping, frozen, actor, seed):
    row, _ = placement.rollout(env, mapping, frozen, seed, actor)
    return row


def development(report, labels, env, mapping, frozen, learner):
    rows = report["development"]
    for label in labels:
        current = rows.setdefault(label, [])
        assert [x["seed"] for x in current] == DEVELOPMENT[:len(current)]
        if len(current) == len(DEVELOPMENT):
            continue
        model = candidate(label, learner)
        for seed in DEVELOPMENT[len(current):]:
            row = evaluate(env, mapping, frozen, model, seed)
            current.append(row)
            progress(report, "development", policy=label, seed=seed, success=row["success"],
                     evaluations=len(current))
    return {label: placement.summary(rows[label]) for label in labels}


def evaluate_test(report, env, mapping, frozen, learner, selection):
    rows = report["test"]
    models = {"vla": None, "actor": candidate(selection, learner)}
    for label in models:
        rows.setdefault(label, [])
    for repeat in range(1, TEST_REPEATS + 1):
        labels = tuple(models) if repeat % 2 else tuple(reversed(models))
        for index, seed in enumerate(TEST):
            offset = (repeat - 1) * len(TEST) + index
            for label in labels:
                if len(rows[label]) > offset:
                    prior = rows[label][offset]
                    assert prior["repeat"] == repeat and prior["seed"] == seed
                    continue
                row = evaluate(env, mapping, frozen, models[label], seed)
                row["repeat"] = repeat
                rows[label].append(row)
                progress(report, "test", policy=label, repeat=repeat, seed=seed, success=row["success"],
                         evaluations=len(rows[label]))
    report["test_summary"] = {label: placement.summary(rows[label]) for label in models}
    report["test_comparison"] = placement.contrasts(rows["vla"], rows["actor"])


def main():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    assert hashlib.sha256(START.read_bytes()).hexdigest() == START_SHA
    assert len(ONLINE + DEVELOPMENT + TEST) == len(set(ONLINE + DEVELOPMENT + TEST))
    report = state_report()
    if report["complete"]:
        print(json.dumps({"phase": report["phase"], "completed_episodes": len(report["train"])}), flush=True)
        return
    env = grasp.SO101BlockBox(ROOT / "menagerie")
    mapping = grasp.JointMapping(json.loads((ROOT / "official_block_follower.json").read_text()))
    begin = perf_counter()
    try:
        frozen = grasp.FrozenPolicy()
        resume_path = LATEST if LATEST.exists() else START
        learner, history = restore(resume_path, frozen.provenance["policy_version"])
        assert [row["seed"] for row in history] == ONLINE[:len(history)]
        report["train"] = history
        if len(history) in SAVE_AT and not path_for(len(history)).exists():
            persist(path_for(len(history)), learner, history, frozen)
        progress(report, "training", resumed_new_episodes=len(history), **learner.status())
        while len(history) < len(ONLINE):
            if len(history) == 32:
                scores = development(report, ("0", "16", "32", "vla"), env, mapping, frozen, learner)
                positive = sum(row["success"] and row["placement_reward_recorded"] == 1.0 for row in history)
                both_worse = all(scores[str(n)]["stable_placement"] < scores["0"]["stable_placement"]
                                 for n in (16, 32))
                report["midpoint_review"] = {
                    "new_positive_terminals": positive, "scores": scores,
                    "failed_before_lift": sum(row["stable_lift_step"] is None and not row["success"] for row in history),
                    "failed_after_lift": sum(row["stable_lift_step"] is not None and not row["success"] for row in history),
                    "gate_not_triggered": sum(row["gate"] is None for row in history),
                    "drops_after_lift_handoff": sum(bool(row["handoff"] and row["handoff"]["first_drop_step"] is not None)
                                                   for row in history),
                    "continue": not (positive < 4 and both_worse),
                }
                progress(report, "midpoint_review", **report["midpoint_review"])
                if not report["midpoint_review"]["continue"]:
                    break
            index = len(history) + 1
            seed = ONLINE[index - 1]
            row, replay = placement.rollout(env, mapping, frozen, seed, learner.actor, learn=True)
            learner.add(replay)
            learner.fit(UPDATES_PER_EPISODE, train_actor=True)
            history.append(row)
            persist(LATEST, learner, history, frozen)
            if index in SAVE_AT:
                persist(path_for(index), learner, history, frozen)
            progress(report, "training", additional_episode=index, seed=seed, success=row["success"],
                     stable_lift=row["stable_lift_step"] is not None, **learner.status())
        report["learner"] = learner.status()
        # Recreate a milestone if the process exited between the latest save and its milestone save.
        if len(history) in SAVE_AT and not path_for(len(history)).exists():
            persist(path_for(len(history)), learner, history, frozen)
        milestones = [index for index in SAVE_AT if path_for(index).exists()]
        labels = ("0", *(str(index) for index in milestones), "vla")
        scores = development(report, labels, env, mapping, frozen, learner)
        best = max(("0", *(str(index) for index in milestones)),
                   key=lambda label: (scores[label]["stable_placement"], -int(label)))
        improves = (best != "0" and scores[best]["stable_placement"] > scores["0"]["stable_placement"]
                    and scores[best]["stable_placement"] > scores["vla"]["stable_placement"])
        report["checkpoints"] = {str(index): str(path_for(index)) for index in milestones}
        report["selection"] = {"candidate": best, "checkpoint": str(path_for(int(best))), "scores": scores,
                               "advance_to_test": improves}
        progress(report, "selection", **report["selection"])
        if improves:
            evaluate_test(report, env, mapping, frozen, learner, best)
        report["complete"] = True
        report["seconds_this_process"] = round(perf_counter() - begin, 1)
        progress(report, "complete", completed_episodes=len(history), learner=report["learner"],
                 test_summary=report.get("test_summary"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
