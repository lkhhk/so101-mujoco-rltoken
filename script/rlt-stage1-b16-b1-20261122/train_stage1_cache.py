"""Train the upstream RLT encoder/decoder using separate B16 and B1 feature caches."""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import lerobot.policies
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "rlt-cache-b16-b1-20261122"
lerobot.policies.__path__.append(str(HERE / "upstream"))
from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTokenDecoder, RLTokenEncoder

UPSTREAM_COMMIT = "e3c1f4517108fdff66d93b68bae3a16fc1cd5dd3"
SEED = 20261122


def open_split(split, batch_size):
    from upstream.train_rlt_stage1_pi05 import MMapPrefixDataset

    root = CACHE / split
    meta = json.loads((root / "meta.json").read_text())
    assert meta["complete"] and meta["batch_size"] == batch_size and meta["image_only"]
    assert (meta["seq_len"], meta["vlm_hidden_dim"]) == (512, 2048)
    return MMapPrefixDataset(root, meta["n_samples"], 512, 2048), meta


def tensors(batch):
    features, mask = batch
    return features.to("cuda", dtype=torch.float32), mask.to("cuda")


def mse(predicted, features, mask):
    error = (predicted.float() - features).square()
    return (error * mask[..., None]).sum() / (mask.sum() * features.shape[-1])


@torch.inference_mode()
def evaluate(encoder, decoder, dataset, batch_size, check_shuffled=False):
    encoder.eval()
    decoder.eval()
    totals = {"mse": 0.0, "first_token_mse": 0.0}
    if check_shuffled:
        totals["mse_shuffled_rl_token"] = 0.0
    count = 0
    for batch in DataLoader(dataset, batch_size=batch_size):
        features, mask = tensors(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            token = encoder(features, mask)
            prediction = decoder(token, features, mask)
            size = len(features)
            totals["mse"] += size * mse(prediction, features, mask).item()
            totals["first_token_mse"] += size * mse(prediction[:, :1], features[:, :1], mask[:, :1]).item()
            if check_shuffled:
                if size == 1:
                    shuffled = torch.zeros_like(token)
                else:
                    shuffled = token.roll(1, dims=0)
                totals["mse_shuffled_rl_token"] += size * mse(decoder(shuffled, features, mask), features, mask).item()
            count += size
    assert all(math.isfinite(value) for value in totals.values())
    return {"frames": count, **{key: value / count for key, value in totals.items()}}


def atomic_json(path, payload):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(payload, indent=2) + "\n")
    pending.replace(path)


def save_checkpoint(path, encoder, decoder, config, optimizer, scheduler, step, val, train_meta):
    pending = path.with_suffix(".tmp")
    torch.save({
        "step": step, "encoder_state_dict": encoder.state_dict(), "decoder_state_dict": decoder.state_dict(),
        "config": vars(config), "rlt_architecture": config.rlt_architecture, "image_only": True,
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "loss": val["mse"], "b1_validation": val, "cache_provenance": train_meta["provenance"],
        "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state(),
        "upstream_commit": UPSTREAM_COMMIT,
    }, pending)
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("high")
    train, train_meta = open_split("train", 16)
    validation, validation_meta = open_split("validation", 1)
    assert train_meta["manifest_sha256"] == validation_meta["manifest_sha256"]
    config = PI05RLTConfig(mode="rlt_training", rlt_lr=1e-4, rlt_warmup_steps=32, rlt_total_steps=args.steps)
    encoder = RLTokenEncoder(config).cuda()
    decoder = RLTokenDecoder(config).cuda()
    params = [*encoder.parameters(), *decoder.parameters()]
    optimizer = torch.optim.AdamW(params, lr=config.rlt_lr, weight_decay=1e-5, fused=True)

    def schedule(step):
        if step < config.rlt_warmup_steps:
            return (step + 1) / config.rlt_warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - config.rlt_warmup_steps)
                                   / (args.steps - config.rlt_warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    order = torch.Generator().manual_seed(SEED)
    batches = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=order, drop_last=True)
    validation_order = np.random.default_rng(SEED).permutation(len(validation)).tolist()
    validate_all = Subset(validation, validation_order)
    validate_subset = Subset(validation, validation_order[:128])
    peaks = {"allocated_mib": 0.0, "reserved_mib": 0.0}
    torch.cuda.reset_peak_memory_stats()
    begin = perf_counter()
    if args.smoke:
        initial = None
    else:
        initial = evaluate(encoder, decoder, validate_all, args.batch_size)
        print(json.dumps({"step": 0, "b1_validation": initial}), flush=True)
    encoder.train()
    decoder.train()
    history = []
    for step, batch in enumerate(batches, start=1):
        features, mask = tensors(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            token = encoder(features, mask)
            loss = mse(decoder(token, features, mask), features, mask)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad) or not torch.isfinite(loss):
            raise FloatingPointError(f"Step {step}: loss={loss.item()} grad={grad.item()}")
        optimizer.step()
        scheduler.step()
        history.append(loss.item())
        peaks = {
            "allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        }
        if step == 1 or step % 32 == 0:
            print(json.dumps({"step": step, "train_mse_mean": float(np.mean(history[-32:])),
                              "grad_norm": grad.item(), "seconds": round(perf_counter() - begin, 1),
                              "peak_cuda": peaks}), flush=True)
        if args.smoke:
            return
        if step == args.steps:
            break
        if step % 128 == 0:
            val = evaluate(encoder, decoder, validate_subset, args.batch_size)
            print(json.dumps({"step": step, "b1_validation_subset": val}), flush=True)
            encoder.train()
            decoder.train()
    assert step == args.steps, "Run schedule exceeds one epoch"
    final = evaluate(encoder, decoder, validate_all, args.batch_size, check_shuffled=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sample, mask = tensors(next(iter(DataLoader(validate_subset, batch_size=1))))
        token = encoder(sample, mask)
        assert token.shape == (1, 2048) and torch.isfinite(token).all()
    checkpoint = HERE / "stage1.pt"
    save_checkpoint(checkpoint, encoder, decoder, config, optimizer, scheduler, step, final, train_meta)
    atomic_json(HERE / "report.json", {
        "upstream_commit": UPSTREAM_COMMIT, "seed": SEED, "steps": step,
        "train_cache_batch": 16, "validation_cache_batch": 1, "training_minibatch": args.batch_size,
        "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
        "decoder_parameters": sum(p.numel() for p in decoder.parameters()),
        "initial_b1_validation": initial, "final_b1_validation": final,
        "b1_encoder_output_shape": list(token.shape),
        "train_loss_first32": float(np.mean(history[:32])), "train_loss_last32": float(np.mean(history[-32:])),
        "peak_cuda": peaks, "seconds": round(perf_counter() - begin, 1), "checkpoint": str(checkpoint),
    })
    print((HERE / "report.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
