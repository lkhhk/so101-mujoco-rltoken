"""Check Stage 1 B16/B1 token agreement and frozen B1 inference in MuJoCo."""
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from train_stage1_cache import CACHE, HERE, PI05RLTConfig, RLTokenEncoder, atomic_json, open_split

sys.path.insert(0, str(CACHE))
from extract_rlt_cache import prefix, processed_batch
from probe_rlt_batches import ADAPTER, BASE, CAMERAS, DATA, MASK, TASK, TOKENIZER
from probe_rlt_batches import load_ira_lora, make_processors, predict_with_features
from evaluate_so101_mujoco_lora import JointMapping
from so101_mujoco_blockbox import SO101BlockBox


def load_encoder():
    checkpoint = torch.load(HERE / "stage1.pt", map_location="cpu", mmap=True, weights_only=False)
    config = PI05RLTConfig(mode="inference")
    saved = checkpoint["config"]
    for name in ("rlt_architecture", "vlm_hidden_dim", "rlt_hidden_dim", "rlt_encoder_layers", "rlt_num_heads"):
        assert saved[name] == getattr(config, name)
    encoder = RLTokenEncoder(config).cuda()
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    return encoder.requires_grad_(False).eval(), checkpoint["step"]


def encode(encoder, features, mask):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        token = encoder(features.to("cuda", dtype=torch.float32), mask.to("cuda"))
    assert torch.isfinite(token).all() and token.shape == (1, 2048)
    return token.float()


def paired(encoder, policy, pre):
    dataset, meta = open_split("validation", 1)
    observations, indices = [], []
    for row in meta["episodes"]:
        folder = DATA / row["directory"]
        with np.load(folder / "motion.npz") as motion:
            states = motion["state"]
        rgb = np.load(folder / "rgb.npy", mmap_mode="r")
        for frame in (row["frames"] // 4, 3 * row["frames"] // 4):
            observations.append(processed_batch(policy, pre, states, rgb, frame, frame + 1, 1))
            indices.append(row["start"] + frame)
    batch = {key: torch.cat([row[key] for row in observations]) for key in observations[0]}
    features, masks = prefix(policy, batch)
    compare = []
    for index, dataset_index in enumerate(indices):
        features_b1, mask_b1 = dataset[dataset_index]
        token_b1 = encode(encoder, features_b1[None], mask_b1[None])
        token_b16 = encode(encoder, features[index:index + 1].to(torch.float16), masks[index:index + 1])
        compare.append((torch.nn.functional.cosine_similarity(token_b1, token_b16).item(),
                        (token_b1 - token_b16).abs().max().item()))
    return {"frames": len(compare), "b1_b16_token_cosine_mean": float(np.mean([row[0] for row in compare])),
            "b1_b16_token_cosine_min": min(row[0] for row in compare),
            "b1_b16_token_max_abs_diff": max(row[1] for row in compare)}


def live(encoder, policy, pre, post):
    root = HERE.parent
    mapping = JointMapping(json.loads((root / "official_block_follower.json").read_text()))
    env = SO101BlockBox(root / "menagerie")
    timings, tokens, windows = [], [], []
    try:
        for seed in (101, 104):
            env.reset(seed=seed)
            random = torch.Generator(device="cuda").manual_seed(seed)
            for _ in range(4):
                t0 = perf_counter()
                obs = {"observation.state": torch.from_numpy(mapping.observation(env.data.qpos[env.position])),
                       "task": TASK}
                for name in CAMERAS:
                    obs[f"observation.images.{name}"] = (
                        torch.from_numpy(env.render_camera(name)).permute(2, 0, 1).float() / 255
                    )
                noise = torch.randn((1, 50, policy.config.max_action_dim), generator=random, device="cuda")
                actions, features, mask = predict_with_features(policy, pre(obs), noise)
                tokens.append(encode(encoder, features, mask).cpu())
                commands = post(actions).detach().float().cpu().numpy()[0]
                assert commands.shape == (50, 6) and np.isfinite(commands).all()
                timings.append((perf_counter() - t0) * 1000)
                for command in commands[:10]:
                    target = np.clip(mapping.radians(command), env.low, env.high)
                    _, _, terminated, truncated, _ = env.step(env.action_for_targets(target))
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    break
            windows.append({"seed": seed, "executed_steps": env.steps})
    finally:
        env.close()
    assert len(timings) == 8 and all(row["executed_steps"] == 40 for row in windows)
    assert not torch.equal(tokens[0], tokens[-1])
    return {"environments": windows, "inference_calls": len(timings), "token_shape": list(tokens[0].shape),
            "control_source": "frozen_vla_reference", "p50_render_vla_encoder_ms": float(np.median(timings)),
            "max_render_vla_encoder_ms": max(timings), "per_call_ms": timings}


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    encoder, step = load_encoder()
    policy, provenance = load_ira_lora(BASE, ADAPTER, device="cuda")
    policy.requires_grad_(False).eval()
    pre, post = make_processors(policy, ADAPTER, TOKENIZER, "cuda")
    pairs = paired(encoder, policy, pre)
    torch.cuda.reset_peak_memory_stats()
    live_result = live(encoder, policy, pre, post)
    report = {"checkpoint_step": step, "adapter_sha256": provenance["adapter_sha256"],
              "heldout_batch_comparison": pairs, "b1_live_simulation": live_result,
              "b1_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    atomic_json(HERE / "b1-smoke.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
