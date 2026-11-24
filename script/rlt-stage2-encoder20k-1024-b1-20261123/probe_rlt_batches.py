"""Capture final image tokens during ordinary PI05 inference on sixteen demo frames."""
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from safetensors.torch import load_file, save_file
import torch

from evaluate_so101_mujoco_lora import BASE_SHA, CAMERAS, TASK
from pi05_rl_runtime import load_ira_lora, make_processors


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent
DATA = ROOT / "scripted-mixed77"
BASE = ROOT.parent / "ira-pi05-v7-full-v2-008000-blockbox-20261117"
ADAPTER = ROOT / "grpo-000140-sft-mixed77-v1/checkpoints/010000"
TOKENIZER = ROOT.parent / "paligemma-tokenizer-35e4f464"
TOKENS = "observation.language.tokens"
MASK = "observation.language.attention_mask"


def sample_frames(policy, pre):
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["task"] == TASK and tuple(manifest["cameras"]) == CAMERAS
    rows = {row["episode"]: row for row in manifest["episodes"]}
    batches, sources = [], []
    for split in ("train", "validation"):
        for episode in manifest[split][:4]:
            row = rows[episode]
            folder = DATA / row["directory"]
            with np.load(folder / "motion.npz") as motion:
                states = motion["state"]
            rgb = np.load(folder / "rgb.npy", mmap_mode="r")
            for frame in (row["frames"] // 4, 3 * row["frames"] // 4):
                obs = {"observation.state": torch.from_numpy(states[frame]), "task": TASK}
                for name, pixels in zip(CAMERAS, rgb[frame]):
                    obs[f"observation.images.{name}"] = (
                        torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255
                    )
                batches.append(pre(obs))
                sources.append({"split": split, "episode": episode, "seed": row["seed"], "frame": frame})
    images = [key for key in policy.config.image_features if key in batches[0]]
    assert len(images) == len(CAMERAS), (images, list(batches[0]))
    keys = [*images, TOKENS, MASK]
    return {key: torch.cat([batch[key] for batch in batches]) for key in keys}, sources


def predict_with_features(policy, batch, noise):
    core = policy.get_base_model().model
    joint = core.paligemma_with_expert
    embed_prefix, forward = core.embed_prefix, joint.forward
    captured = {}
    text_width = batch[TOKENS].shape[1]

    def capture_mask(*args, **kwargs):
        embeds, pad, attention = embed_prefix(*args, **kwargs)
        image_width = embeds.shape[1] - text_width
        captured["mask"] = pad[:, :image_width]
        return embeds, pad, attention

    def capture_states(*args, **kwargs):
        outputs, cache = forward(*args, **kwargs)
        if outputs[0] is not None:
            assert "features" not in captured
            width = captured["mask"].shape[1]
            captured["features"] = outputs[0][:, :width].detach().contiguous()
        return outputs, cache

    with patch.object(core, "embed_prefix", capture_mask), patch.object(joint, "forward", capture_states):
        actions = policy.predict_action_chunk(batch, noise=noise)
    return actions, captured["features"], captured["mask"]


def measure(batch_size, policy, post, batch, noise):
    warmup = {key: value[:batch_size] for key, value in batch.items()}
    predict_with_features(policy, warmup, noise[:batch_size].clone())
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    max_action_delta, max_physical_delta = 0.0, 0.0
    feature_chunks, mask_chunks, action_chunks = [], [], []
    for offset in range(0, len(noise), batch_size):
        section = slice(offset, offset + batch_size)
        obs = {key: value[section] for key, value in batch.items()}
        base = policy.predict_action_chunk(obs, noise=noise[section].clone())
        action, features, mask = predict_with_features(policy, obs, noise[section].clone())
        physical = post(action)
        max_action_delta = max(max_action_delta, (base - action).abs().max().item())
        max_physical_delta = max(max_physical_delta, (post(base) - physical).abs().max().item())
        assert torch.equal(base, action) and torch.isfinite(physical).all()
        assert features.ndim == 3 and features.shape[:2] == mask.shape
        assert torch.isfinite(features).all() and mask.any(dim=1).all()
        assert base.shape == (len(action), 50, 6)
        feature_chunks.append(features.cpu())
        mask_chunks.append(mask.cpu())
        action_chunks.append(action.cpu())
    torch.cuda.synchronize()
    result = {
        "batch_size": batch_size,
        "seconds_for_baseline_plus_capture": round(perf_counter() - start, 3),
        "max_reference_action_delta_normalized": max_action_delta,
        "max_reference_action_delta_physical": max_physical_delta,
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }
    print(json.dumps(result), flush=True)
    return result, {
        "image_tokens": torch.cat(feature_chunks).contiguous(),
        "image_mask": torch.cat(mask_chunks).contiguous(),
        "normalized_reference_actions": torch.cat(action_chunks).contiguous(),
    }


def compare(baseline, candidate):
    first = baseline["image_tokens"].float()
    second = candidate["image_tokens"].float()
    mask = candidate["image_mask"]
    assert torch.equal(baseline["image_mask"], mask)
    delta = (second - first).abs()[mask]
    cosine = torch.nn.functional.cosine_similarity(first[mask], second[mask], dim=-1)
    actions = (baseline["normalized_reference_actions"] - candidate["normalized_reference_actions"]).abs()
    return {
        "feature_abs_mean": delta.mean().item(),
        "feature_abs_max": delta.max().item(),
        "feature_abs_fraction_over_one": (delta > 1).float().mean().item(),
        "token_cosine_mean": cosine.mean().item(),
        "token_cosine_min": cosine.min().item(),
        "normalized_reference_action_abs_max": actions.max().item(),
    }


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    policy, provenance = load_ira_lora(BASE, ADAPTER, device="cuda")
    assert provenance["base_sha256"] == BASE_SHA
    policy.requires_grad_(False).eval()
    print("Strict checkpoint load passed", flush=True)
    pre, post = make_processors(policy, ADAPTER, TOKENIZER, "cuda")
    batch, sources = sample_frames(policy, pre)
    assert len(sources) == 16
    noise = torch.randn(
        (len(sources), 50, policy.config.max_action_dim), device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(20261122),
    )
    runs, features = {}, {}
    for batch_size in (1, 8, 16):
        try:
            runs[batch_size], features[batch_size] = measure(batch_size, policy, post, batch, noise)
        except torch.cuda.OutOfMemoryError as error:
            runs[batch_size] = {"batch_size": batch_size, "oom": str(error)}
            print(json.dumps(runs[batch_size]), flush=True)
            continue

    baseline = features[1]
    comparisons = {}
    for batch_size in (8, 16):
        if batch_size in features:
            comparisons[f"B{batch_size}_vs_B1"] = compare(baseline, features[batch_size])
    if 8 in features and 16 in features:
        comparisons["B16_vs_B8"] = compare(features[8], features[16])
    report = {
        "sources": sources,
        "provenance": provenance,
        "image_token_shape": list(baseline["image_tokens"].shape),
        "image_token_dtype": str(baseline["image_tokens"].dtype),
        "runs": runs,
        "comparisons": comparisons,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("sources", "provenance")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
