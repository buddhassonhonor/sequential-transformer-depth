"""Post-hoc branch-utilization evaluation for completed PAR24_2 checkpoints.

This script intentionally performs inference only.  It uses the exact validation
sampling rule used by training (NumPy seed 9999) so full-model PPL can be checked
against the terminal training evaluation before interpreting branch ablations.
"""

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import psutil
import torch

from data import TokenDataset
from model import GPT


def gpu_health():
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=5
        ).strip()
        temp, memory = (float(x.strip()) for x in text.split(","))
        return {"temperature_c": temp, "memory_used_mb": memory}
    except Exception:
        return {"temperature_c": None, "memory_used_mb": None}


@torch.no_grad()
def evaluate(model, dataset, mode, batches, batch_size, random_seed=9999):
    """Evaluate one ablation mode on an identical deterministic batch sequence."""
    model.eval()
    saved_state = np.random.get_state()
    np.random.seed(random_seed)
    losses = []
    for _ in range(batches):
        x, y = dataset.get_batch(batch_size=batch_size, device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, targets=y, ablation_mode=mode)
        losses.append(float(loss.item()))
    np.random.set_state(saved_state)
    loss = float(np.mean(losses))
    return {"loss": loss, "ppl": float(math.exp(loss)), "batches": batches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="D:/data/tinystories_10m")
    parser.add_argument("--output-dir", default="results_stage5")
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation protocol.")
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    config = checkpoint["config"]
    if config.model_type.lower() != "par24_2":
        raise ValueError(f"Expected PAR24_2 checkpoint, found {config.model_type}")

    model = GPT(config).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = TokenDataset(os.path.join(args.data_dir, "val.bin"), block_size=config.block_size)
    torch.cuda.reset_peak_memory_stats()

    outcomes = {}
    for mode in ("full", "branch1", "branch2", "random_single"):
        outcomes[mode] = evaluate(model, dataset, mode, args.batches, args.batch_size)
        outcomes[mode]["ppl_delta_from_full"] = outcomes[mode]["ppl"] - outcomes["full"]["ppl"]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_tokens": checkpoint.get("tokens_done"),
        "model": config.model_type,
        "validation_protocol": {
            "numpy_seed": 9999, "batches": args.batches, "batch_size": args.batch_size,
            "precision": "BF16 autocast", "dataset": str(Path(args.data_dir).resolve()),
        },
        "outcomes": outcomes,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "host_ram_used_gb": round(psutil.virtual_memory().used / 1024**3, 2),
        "gpu_after": gpu_health(),
    }
    stem = Path(args.checkpoint).stem.replace("_50M", "")
    output_path = Path(args.output_dir) / f"{stem}_branch_ablation.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
