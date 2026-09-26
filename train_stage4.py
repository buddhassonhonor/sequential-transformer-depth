"""
train_stage4.py - Stage 4A Training Script: 24-Block Sequential-Depth Scaling Pilot.
Supports SEQ24 (depth=24) and PAR24_2 (depth=12, 12 stages x 2 branches).

Strictly adheres to user specifications:
- 50M-token cosine schedule created from the start (warmup=2%, decay horizon=50M).
- Phase 1 stop at 20M tokens (evaluating 2M, 5M, 10M, 15M, 20M).
- Resume support from 20M checkpoint to 50M tokens (evaluating 30M, 40M, 50M).
- Only saves checkpoints for 20M, 50M, and best (to protect SSD).
- Temperature and VRAM monitoring every minute / eval step.
"""

import os
import sys
import time
import math
import json
import argparse
import subprocess
import psutil
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from data import TokenDataset

# ── Milestones ───────────────────────────────────────────────────────────────
STAGE4_MILESTONE_TOKENS = [
    2_000_000,
    5_000_000,
    10_000_000,
    15_000_000,
    20_000_000,
    30_000_000,
    40_000_000,
    50_000_000,
]

def get_gpu_health():
    """Query nvidia-smi for temperature, utilization, and memory."""
    try:
        res = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        temp, util, mem = [float(x.strip()) for x in res.stdout.strip().split(',')]
        return temp, util, mem
    except Exception:
        return -1.0, -1.0, -1.0

# ── LR schedule ──────────────────────────────────────────────────────────────
def get_lr(step, total_steps, warmup_steps, peak_lr, min_lr):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)

# ── Validation (Deterministic subset) ────────────────────────────────────────
@torch.no_grad()
def estimate_val_loss(model, val_dataset, num_batches=40, batch_size=32, device='cuda', seed=9999):
    model.eval()
    losses = []
    rng_state = np.random.get_state()
    np.random.seed(seed)
    
    for _ in range(num_batches):
        x, y = val_dataset.get_batch(batch_size=batch_size, device=device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
        losses.append(loss.item())
        
    np.random.set_state(rng_state)
    model.train()
    mean_loss = float(np.mean(losses))
    ppl = float(math.exp(mean_loss)) if mean_loss < 20 else float('inf')
    return mean_loss, ppl

# ── Build model config ────────────────────────────────────────────────────────
def build_config(m_type: str, d_model: int, n_head: int, block_size: int, dropout: float):
    m_type = m_type.lower()
    if m_type in ('seq24', 'par24_2'):
        n_layer = 24
        d_ff = 0
    elif m_type in ('seq12', 'par12_2', 'par12_3'):
        n_layer = 12
        d_ff = 0
    elif m_type in ('seq6', 'seq6_wide'):
        n_layer = 6
        d_ff = 0
    else:
        raise ValueError(f"Unknown model type: {m_type}")

    return GPTConfig(
        block_size=block_size,
        vocab_size=50304,
        n_layer=n_layer,
        n_head=n_head,
        d_model=d_model,
        d_ff=d_ff,
        dropout=dropout,
        model_type=m_type,
        weight_tying=True,
    )

# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Data
    train_bin = os.path.join(args.data_dir, "train.bin")
    val_bin   = os.path.join(args.data_dir, "val.bin")
    assert os.path.exists(train_bin), f"Train bin not found: {train_bin}"
    assert os.path.exists(val_bin),   f"Val bin not found: {val_bin}"
    train_dataset = TokenDataset(train_bin, block_size=args.block_size)
    val_dataset   = TokenDataset(val_bin,   block_size=args.block_size)

    # Model
    config = build_config(args.model, args.d_model, args.n_head, args.block_size, args.dropout)
    model  = GPT(config).to(device)
    param_count    = sum(p.numel() for p in model.parameters())
    non_emb_params = model.get_num_params(non_embedding=True)
    m_label = args.model.upper()
    print(f"[{m_label} | seed={args.seed}] {param_count:,} total params | {non_emb_params:,} non-emb params")

    # Optimizer
    decay_params   = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() <  2]
    optimizer = torch.optim.AdamW(
        [{'params': decay_params,   'weight_decay': args.weight_decay},
         {'params': nodecay_params, 'weight_decay': 0.0}],
        lr=args.peak_lr, betas=(0.9, 0.95), eps=1e-8
    )

    # Full 50M schedule
    tokens_per_step   = args.batch_size * args.block_size
    schedule_steps    = math.ceil(args.total_schedule_tokens / tokens_per_step)
    warmup_steps      = max(1, int(schedule_steps * 0.02))
    target_stop_steps = math.ceil(args.max_tokens / tokens_per_step)
    
    print(f"[{m_label}] Schedule horizon = {schedule_steps} steps ({args.total_schedule_tokens/1e6:.1f}M tok), warmup={warmup_steps}")
    print(f"[{m_label}] This run stop step = {target_stop_steps} steps ({args.max_tokens/1e6:.1f}M tok)")

    # Milestones in terms of steps
    milestone_steps = {}
    for mtok in STAGE4_MILESTONE_TOKENS:
        m_step = math.ceil(mtok / tokens_per_step)
        if m_step <= schedule_steps:
            milestone_steps[m_step] = mtok

    run_id    = f"{args.model.lower()}_seed{args.seed}"
    log_jsonl = os.path.join(args.output_dir, f"{run_id}.jsonl")
    log_csv   = os.path.join(args.output_dir, f"{run_id}.csv")
    best_ckpt = os.path.join(args.ckpt_dir, f"{run_id}_best.pt")
    milestone_json = os.path.join(args.output_dir, f"{run_id}_milestones.json")

    start_step = 0
    best_val_loss = float('inf')
    history = []
    milestone_results = []

    # Resume if requested
    if args.resume_ckpt:
        print(f"[{m_label}] Resuming from checkpoint: {args.resume_ckpt}")
        ckpt_data = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data['model_state_dict'])
        if 'optimizer_state_dict' in ckpt_data and ckpt_data['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
        start_step = ckpt_data.get('step', 0)
        best_val_loss = ckpt_data.get('best_val_loss', ckpt_data.get('val_loss', float('inf')))
        print(f"[{m_label}] Resumed at step {start_step} ({start_step * tokens_per_step / 1e6:.2f}M tok), best_val_loss={best_val_loss:.4f}")
        
        # Load existing milestones if available
        if os.path.exists(milestone_json):
            with open(milestone_json, "r") as fp:
                old_meta = json.load(fp)
                milestone_results = old_meta.get("milestones", [])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Initial val check if starting from scratch
    if start_step == 0:
        val_loss_init, val_ppl_init = estimate_val_loss(
            model, val_dataset, num_batches=30, batch_size=args.batch_size, device=device)
        print(f"Step 0 | Initial Val Loss={val_loss_init:.4f} PPL={val_ppl_init:.2f}")

    model.train()
    t_start = time.perf_counter()
    last_temp_check_time = time.time()

    for step in range(start_step, target_stop_steps):
        lr = get_lr(step, schedule_steps, warmup_steps, args.peak_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x, y = train_dataset.get_batch(batch_size=args.batch_size, device=device)

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(x, targets=y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        tokens_per_sec = tokens_per_step / (t1 - t0) if (t1 - t0) > 0 else 0.0
        current_step = step + 1
        is_last_step = (current_step == target_stop_steps)
        is_milestone = current_step in milestone_steps
        do_eval = (current_step % args.eval_interval == 0) or is_last_step or is_milestone

        # Periodic health check every 60s
        now = time.time()
        if now - last_temp_check_time > 60.0:
            gpu_temp, gpu_util, _ = get_gpu_health()
            last_temp_check_time = now
            if gpu_temp > 87.0:
                print(f"\n[ALERT] GPU Temperature reached {gpu_temp}°C (>87°C threshold)! Pausing for 30s to cool down...")
                time.sleep(30.0)

        if do_eval:
            val_loss, val_ppl = estimate_val_loss(
                model, val_dataset, num_batches=40, batch_size=args.batch_size, device=device)
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
            tokens_done = current_step * tokens_per_step
            gpu_temp, gpu_util, _ = get_gpu_health()

            log_entry = {
                "step":              current_step,
                "tokens_processed":  tokens_done,
                "train_loss":        round(loss.item(), 4),
                "val_loss":          round(val_loss, 4),
                "val_perplexity":    round(val_ppl, 2),
                "lr":                round(lr, 6),
                "grad_norm":         round(grad_norm, 4),
                "tokens_per_second": round(tokens_per_sec, 1),
                "peak_vram_mb":      round(peak_vram, 2),
                "gpu_temp_c":        round(gpu_temp, 1),
                "gpu_util_pct":      round(gpu_util, 1),
                "parameter_count":   param_count,
                "non_emb_params":    non_emb_params,
            }
            history.append(log_entry)
            with open(log_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"Step {current_step:5d}/{target_stop_steps} ({tokens_done/1e6:5.1f}M tok) | "
                  f"TLoss={loss.item():.4f} VLoss={val_loss:.4f} PPL={val_ppl:5.2f} | "
                  f"tok/s={tokens_per_sec:.0f} VRAM={peak_vram:.0f}MB Temp={gpu_temp:.0f}°C")

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'step': current_step,
                    'tokens_done': tokens_done,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'val_loss': val_loss,
                    'val_ppl': val_ppl,
                    'best_val_loss': best_val_loss,
                    'non_emb_params': non_emb_params,
                }, best_ckpt)

            # Record milestone
            if is_milestone:
                mtok = milestone_steps[current_step]
                milestone_entry = {
                    'tokens_M': mtok / 1e6,
                    'step': current_step,
                    'val_loss': val_loss,
                    'val_ppl': val_ppl,
                    'tokens_per_second': round(tokens_per_sec, 1),
                    'peak_vram_mb': round(peak_vram, 1),
                    'gpu_temp_c': gpu_temp
                }
                # Update if already exists or append
                milestone_results = [m for m in milestone_results if abs(m['tokens_M'] - mtok/1e6) > 0.1]
                milestone_results.append(milestone_entry)
                
                # Checkpoint saving policy per Section 24: only keep 20M and 50M
                if abs(mtok - 20_000_000) < 100_000 or abs(mtok - 50_000_000) < 100_000:
                    tag = "20M" if abs(mtok - 20_000_000) < 100_000 else "50M"
                    m_ckpt = os.path.join(args.ckpt_dir, f"{run_id}_{tag}.pt")
                    torch.save({
                        'step': current_step,
                        'tokens_done': tokens_done,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                        'val_loss': val_loss,
                        'val_ppl': val_ppl,
                        'best_val_loss': best_val_loss,
                        'non_emb_params': non_emb_params,
                    }, m_ckpt)
                    print(f"  >>> SAVED MILESTONE CHECKPOINT ({tag}): {m_ckpt}")
                else:
                    print(f"  >>> MILESTONE {mtok/1e6:.0f}M | PPL={val_ppl:.2f}")

    # Save CSV
    df = pd.DataFrame(history)
    if os.path.exists(log_csv) and args.resume_ckpt:
        df_old = pd.read_csv(log_csv)
        df = pd.concat([df_old, df], ignore_index=True).drop_duplicates(subset=['step'])
    df.to_csv(log_csv, index=False)

    # Save milestone summary JSON
    with open(milestone_json, "w") as f:
        json.dump({
            "model": args.model,
            "seed": args.seed,
            "non_emb_params": non_emb_params,
            "best_val_loss": best_val_loss,
            "best_val_ppl": math.exp(best_val_loss) if best_val_loss < 20 else float('inf'),
            "milestones": sorted(milestone_results, key=lambda x: x['tokens_M']),
        }, f, indent=2)

    duration = time.perf_counter() - t_start
    print(f"[{m_label}] Done in {duration:.1f}s | Best Val Loss={best_val_loss:.4f} | Logs: {log_csv}")
    return log_csv, best_ckpt, milestone_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4A Training: 24-Block Sequential-Depth Scaling Pilot")
    parser.add_argument("--model", type=str, required=True, choices=["seq24", "par24_2"])
    parser.add_argument("--seed",                  type=int,   default=42)
    parser.add_argument("--total_schedule_tokens", type=int,   default=50_000_000)
    parser.add_argument("--max_tokens",            type=int,   default=20_000_000)
    parser.add_argument("--resume_ckpt",           type=str,   default=None)
    parser.add_argument("--batch_size",            type=int,   default=32)
    parser.add_argument("--block_size",            type=int,   default=256)
    parser.add_argument("--d_model",               type=int,   default=384)
    parser.add_argument("--n_head",                type=int,   default=6)
    parser.add_argument("--dropout",               type=float, default=0.1)
    parser.add_argument("--peak_lr",               type=float, default=3e-4)
    parser.add_argument("--min_lr",                type=float, default=3e-5)
    parser.add_argument("--weight_decay",          type=float, default=0.1)
    parser.add_argument("--grad_clip",             type=float, default=1.0)
    parser.add_argument("--eval_interval",         type=int,   default=200)
    parser.add_argument("--data_dir",              type=str,   default="D:/data/tinystories_10m")
    parser.add_argument("--output_dir",            type=str,   default="logs_stage4")
    parser.add_argument("--ckpt_dir",              type=str,   default="checkpoints_stage4")
    args = parser.parse_args()
    train(args)
