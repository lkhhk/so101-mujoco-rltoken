"""Build Stage 1 mmap caches with B16 training and B1 deployment validation."""
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
import numpy as np
import torch

from probe_rlt_batches import (
    ADAPTER, BASE, BASE_SHA, CAMERAS, DATA, MASK, TASK, TOKENIZER, TOKENS,
    load_ira_lora, make_processors, predict_with_features, sample_frames,
)


OUTPUT = Path(__file__).resolve().parent
SHAPE = (512, 2048)
SPLITS = {"train": 16, "validation": 1}


def write_json(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, indent=2) + "\n")
    pending.replace(path)


def prefix(policy, batch):
    core = policy.get_base_model().model
    joint = core.paligemma_with_expert
    images, masks = policy._preprocess_images(batch)
    tokens = batch[TOKENS]
    embeds, pad, attention = core.embed_prefix(images, masks, tokens, batch[MASK])
    attention = core._prepare_attention_masks_4d(make_att_2d_masks(pad, attention))
    joint.paligemma.language_model.config._attn_implementation = "eager"
    (features, _), _ = joint.forward(
        attention_mask=attention, position_ids=pad.long().cumsum(1) - 1,
        past_key_values=None, inputs_embeds=[embeds, None], use_cache=True,
    )
    width = embeds.shape[1] - tokens.shape[1]
    return features[:, :width], pad[:, :width]


def parity(policy, pre):
    batch, sources = sample_frames(policy, pre)
    noise = torch.zeros((len(sources), 50, policy.config.max_action_dim), device="cuda")
    for batch_size in SPLITS.values():
        obs = {key: values[:batch_size] for key, values in batch.items()}
        expected_action, expected, expected_mask = predict_with_features(policy, obs, noise[:batch_size])
        features, mask = prefix(policy, obs)
        assert expected_action.shape == (batch_size, 50, 6)
        assert torch.equal(expected, features) and torch.equal(expected_mask, mask), batch_size
        assert features.shape == (batch_size, *SHAPE) and features.dtype == torch.bfloat16
        assert torch.equal(features, features.to(torch.float16).to(torch.bfloat16))
        print(json.dumps({"parity_batch_size": batch_size, "exact_feature_and_mask": True}), flush=True)
    return [16, 1]


def processed_batch(policy, pre, states, images, start, stop, batch_size):
    frames = [*range(start, stop), *([stop - 1] * (batch_size - stop + start))]
    processed = []
    for index in frames:
        obs = {"observation.state": torch.from_numpy(states[index]), "task": TASK}
        for name, pixels in zip(CAMERAS, images[index]):
            obs[f"observation.images.{name}"] = (
                torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255
            )
        processed.append(pre(obs))
    image_keys = [key for key in policy.config.image_features if key in processed[0]]
    assert len(image_keys) == len(CAMERAS)
    return {
        key: torch.cat([item[key] for item in processed])
        for key in [*image_keys, TOKENS, MASK]
    }


def metadata(manifest, split, provenance, manifest_sha, policy):
    rows = {row["episode"]: row for row in manifest["episodes"]}
    offset = 0
    episodes = []
    for episode in manifest[split]:
        row = rows[episode]
        episodes.append({"episode": episode, "directory": row["directory"],
                         "seed": row["seed"], "frames": row["frames"], "start": offset})
        offset += row["frames"]
    return {
        "schema_version": 2, "n_samples": offset, "has_mask": True, "dtype": "float16",
        "image_only": True, "embedding_stage": "post_transformer", "language_positions_dropped": True,
        "seq_len": SHAPE[0], "vlm_hidden_dim": SHAPE[1], "tokens_per_camera": SHAPE[0] // len(CAMERAS),
        "pi05_path": str(ADAPTER), "dataset_repo_id": DATA.name,
        "checkpoint_image_features": list(policy.config.image_features),
        "split": split, "batch_size": SPLITS[split], "manifest_sha256": manifest_sha,
        "provenance": provenance, "episodes": episodes,
    }


def remaining_bytes(meta):
    folder = OUTPUT / meta["split"]
    previous = folder / "meta.json"
    if previous.exists():
        progress = json.loads(previous.read_text())
        assert all(progress.get(key) == value for key, value in meta.items()), "Cache identity changed"
        if progress["complete"]:
            return 0
    required = meta["n_samples"] * SHAPE[0] * (SHAPE[1] * 2 + 1)
    existing = sum((folder / name).stat().st_blocks * 512 for name in ("prefix_out.mmap", "prefix_mask.mmap")
                   if (folder / name).exists())
    return max(0, required - existing)


def write_split(policy, pre, meta):
    split = meta["split"]
    folder = OUTPUT / split
    folder.mkdir(exist_ok=True)
    status_path = folder / "meta.json"
    progress = json.loads(status_path.read_text()) if status_path.exists() else {}
    done = progress.get("episodes_written", 0)
    expected_frames = sum(row["frames"] for row in meta["episodes"][:done])
    assert progress.get("samples_written", 0) == expected_frames
    n_samples, batch_size = meta["n_samples"], meta["batch_size"]
    prefix_path, mask_path = folder / "prefix_out.mmap", folder / "prefix_mask.mmap"
    if done:
        assert prefix_path.stat().st_size == n_samples * SHAPE[0] * SHAPE[1] * 2
        assert mask_path.stat().st_size == n_samples * SHAPE[0]
    if progress.get("complete"):
        assert done == len(meta["episodes"]) and expected_frames == n_samples
        return progress
    mode = "r+" if done else "w+"
    embeddings = np.memmap(prefix_path, dtype=np.float16, mode=mode, shape=(n_samples, *SHAPE))
    masks = np.memmap(mask_path, dtype=np.bool_, mode=mode, shape=(n_samples, SHAPE[0]))
    if done == 0:
        write_json(status_path, {**meta, "complete": False, "episodes_written": 0, "samples_written": 0})
    start_time = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for episode_index, row in enumerate(meta["episodes"][done:], start=done + 1):
        episode = DATA / row["directory"]
        with np.load(episode / "motion.npz") as motion:
            states = motion["state"]
        images = np.load(episode / "rgb.npy", mmap_mode="r")
        frames, offset = row["frames"], row["start"]
        assert states.shape == (frames, 6) and np.isfinite(states).all()
        assert images.shape == (frames, len(CAMERAS), 240, 320, 3) and images.dtype == np.uint8
        for start in range(0, frames, batch_size):
            stop = min(start + batch_size, frames)
            batch = processed_batch(policy, pre, states, images, start, stop, batch_size)
            features, mask = prefix(policy, batch)
            assert features.shape == (batch_size, *SHAPE)
            assert mask.shape == (batch_size, SHAPE[0]) and mask.dtype == torch.bool
            cached = features[:stop - start].to(torch.float16).cpu()
            cached_mask = mask[:stop - start].cpu()
            assert torch.isfinite(cached).all() and cached_mask.all()
            embeddings[offset + start:offset + stop] = cached.numpy()
            masks[offset + start:offset + stop] = cached_mask.numpy()
        embeddings.flush()
        masks.flush()
        total = offset + frames
        update = {
            "split": split, "batch_size": batch_size, "episodes_written": episode_index,
            "samples_written": total, "n_samples": n_samples,
            "elapsed_seconds": round(perf_counter() - start_time, 1),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        }
        write_json(status_path, {**meta, **update, "complete": total == n_samples})
        print(json.dumps(update), flush=True)
    del embeddings, masks
    return json.loads(status_path.read_text())


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    data = (DATA / "manifest.json").read_bytes()
    manifest = json.loads(data)
    assert manifest["task"] == TASK and tuple(manifest["cameras"]) == CAMERAS
    assert len(manifest["train"]) == 77 and len(manifest["validation"]) == 8
    assert not set(manifest["train"]) & set(manifest["validation"])
    policy, provenance = load_ira_lora(BASE, ADAPTER, device="cuda")
    assert provenance["base_sha256"] == BASE_SHA
    policy.requires_grad_(False).eval()
    pre, _ = make_processors(policy, ADAPTER, TOKENIZER, "cuda")
    checked = parity(policy, pre)
    configs = [metadata(manifest, split, provenance, hashlib.sha256(data).hexdigest(), policy) for split in SPLITS]
    required = sum(remaining_bytes(config) for config in configs)
    free = shutil.disk_usage(OUTPUT).free
    if free < required + 2 * 2**30:
        raise RuntimeError(f"Cache needs {required / 2**30:.2f} GiB plus 2 GiB free, have {free / 2**30:.2f} GiB")
    print(json.dumps({"cache_bytes_to_write": required, "initial_free_bytes": free}), flush=True)
    results = [write_split(policy, pre, config) for config in configs]
    assert all(row["complete"] for row in results)
    fields = ("split", "batch_size", "n_samples", "episodes_written", "elapsed_seconds", "peak_allocated_mib")
    write_json(OUTPUT / "report.json", {
        "complete": True,
        "prefix_parity_batches": checked,
        "cache_layout": "RL-Token-Pi05 Stage 1 mmap v2",
        "splits": [{key: row[key] for key in fields} for row in results],
        "remaining_bytes": shutil.disk_usage(OUTPUT).free,
    })


if __name__ == "__main__":
    main()
