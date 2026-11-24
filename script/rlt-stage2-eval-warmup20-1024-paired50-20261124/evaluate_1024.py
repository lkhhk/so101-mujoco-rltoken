import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REFERENCE = ROOT.parent / "results" / "paired-baselines.json"
sys.path.insert(0, str(ROOT / "rlt-stage2-encoder20k-1024-b1-20261123"))
import continue_actor_with_new_encoder as stage

grasp, placement = stage.grasp, stage.base.placement
CHECKPOINT = ROOT / "rlt-stage2-new-encoder-640-warmup20-b1-20261124" / "episode-1024.pt"
REPORT = HERE / "report.json"
BASELINES = ("vla_sft", "new_encoder_fresh_640")


def main():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    ref = json.loads(REFERENCE.read_text())
    seeds = list(range(65000, 65050))
    assert ref["complete"] and ref["settings"]["seeds"] == seeds
    assert all([r["seed"] for r in ref["policies"][k]] == seeds for k in BASELINES)
    saved = torch.load(CHECKPOINT, map_location="cpu", mmap=True, weights_only=False)
    history = saved["report"]
    assert history["episodes_done"] == 1024 and history["complete"]
    assert history["continuation_start"] == 640 and len(history["continuation_warmup"]) == 20
    assert history["continuation_critic_warmup_updates"] == 300
    assert history["encoder_total_step"] == 20512
    assert saved["policy_batch_size"] == ref["settings"]["batch_size"] == 1
    assert saved["chunk_steps"] == ref["settings"]["chunk_steps"] == grasp.CHUNK
    assert not set(seeds) & {r["seed"] for key in ("online", "warmup", "continuation_warmup")
                             for r in history[key]}
    encoder_sha = stage.file_sha256(stage.ENCODER)
    assert encoder_sha == history["encoder_sha256"]
    settings = {"seeds": seeds, "batch_size": 1, "chunk_steps": grasp.CHUNK,
                "checkpoint": str(CHECKPOINT), "checkpoint_sha256": stage.file_sha256(CHECKPOINT),
                "encoder": str(stage.ENCODER), "encoder_sha256": encoder_sha,
                "baseline_report": str(REFERENCE)}
    report = {"settings": settings, "complete": False, "episodes": []}
    if REPORT.exists():
        report = json.loads(REPORT.read_text())
        assert report["settings"] == settings
        if report["complete"]:
            return
    rows = report["episodes"]
    assert [row["seed"] for row in rows] == seeds[:len(rows)]
    with patch.object(grasp, "load_encoder", stage.load_encoder):
        frozen = grasp.FrozenPolicy()
    assert frozen.encoder_step == 20512
    assert (saved["vla_provenance"]["policy_version"] == frozen.provenance["policy_version"]
            == ref["settings"]["vla_provenance"]["policy_version"])
    actor = grasp.RLTChunkActor(grasp.PI05RLTConfig(mode="online_rl")).cuda()
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    actor.requires_grad_(False).eval()
    del saved
    env = grasp.SO101BlockBox(ROOT / "menagerie")
    mapping_path = ROOT.parent / "config" / "official_block_follower.json"
    mapping = grasp.JointMapping(json.loads(mapping_path.read_text()))
    try:
        for seed in seeds[len(rows):]:
            row, _ = placement.rollout(env, mapping, frozen, seed, actor)
            rows.append(row)
            grasp.save(REPORT, report)
            print(json.dumps({"done": len(rows), "seed": seed, "success": row["success"],
                              "stable_lift": row["stable_lift_step"] is not None,
                              "steps": row["steps"], "seconds": row["seconds"]}), flush=True)
    finally:
        env.close()
    report["summary"] = placement.summary(rows)
    report["paired_vs_baselines"] = {k: placement.contrasts(ref["policies"][k], rows) for k in BASELINES}
    report["complete"] = True
    grasp.save(REPORT, report)
    print(json.dumps({"summary": report["summary"], "paired": report["paired_vs_baselines"]}), flush=True)


if __name__ == "__main__":
    main()
