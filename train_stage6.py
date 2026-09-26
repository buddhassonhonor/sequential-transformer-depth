"""
train_stage6.py - Stage 6 Training Script: Cross-Corpus Generalization & 3x Compression on WikiText-103.
Models supported: SEQ24, PAR24_2, PAR24_3, SEQ8_WIDE.

Key principles:
- Official WikiText-103 raw dataset tokenized with GPT-2 BPE (117.9M tokens).
- Fixed 50M-token cosine schedule (warmup=2%, decay to 3e-5).
- Guaranteed identical training token sequences across models via deterministic RandomState(seed).
- Guaranteed identical validation batches across models via pre-generated validation_indices.npy.
- Checkpoints saved for best and 50M.
- Temperature and VRAM monitoring every minute.
"""

import os
import sys
import time
import math
import json
import argparse
import subprocess
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

# ── Milestones ───────────────────────────────────────────────────────────────
STAGE6_MILESTONE_TOKENS = [
    5_000_000,
    10_000_000,
    20_000_000,
    30_000_000,
    40_000_000,
    50_000_000,
]

def get_gpu_health():
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

# ── Deterministic Validation ─────────────────────────────────────────────────
@torch.no_grad()
def evaluate_deterministic_val(model, val_data, val_indices, block_size=256, device='cuda'):
    model.eval()
    losses = []
    num_batches, batch_size = val_indices.shape
    
    for b_idx in range(num_batches):
        batch_offsets = val_indices[b_idx]
        x_list = [val_data[i : i + block_size].astype(np.int64) for i in batch_offsets]
        y_list = [val_data[i + 1 : i + 1 + block_size].astype(np.int64) for i in batch_offsets]
        x = torch.from_numpy(np.stack(x_list))
        y = torch.from_numpy(np.stack(y_list))
        if 'cuda' in str(device) and torch.cuda.is_available():
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
            
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
        losses.append(loss.item())
        
    model.train()
    mean_loss = float(np.mean(losses))
    ppl = float(math.exp(mean_loss)) if mean_loss < 20 else float('inf')
    return mean_loss, ppl

# ── Build model config ────────────────────────────────────────────────────────
def build_config(m_type: str, d_model: int, n_head: int, block_size: int, dropout: float):
    m_type = m_type.lower()
    if m_type in ('seq24', 'par24_2', 'par24_3'):
        n_layer = 24
        d_ff = 0
    elif m_type == 'seq8_wide':
        n_layer = 8
        d_ff = 6148
    elif m_type in ('seq12', 'par12_2', 'par12_3'):
        n_layer = 12
        d_ff = 0
    elif m_type == 'seq6':
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

    # Data paths
    train_bin = os.path.join(args.data_dir, "train.bin")
    val_bin   = os.path.join(args.data_dir, "val.bin")
    val_idx_file = os.path.join(args.data_dir, "validation_indices.npy")
    assert os.path.exists(train_bin), f"Train bin not found: {train_bin}"
    assert os.path.exists(val_bin),   f"Val bin not found: {val_bin}"
    assert os.path.exists(val_idx_file), f"Val indices not found: {val_idx_file}"

    train_data = np.memmap(train_bin, dtype=np.uint16, mode='r')
    val_data   = np.memmap(val_bin,   dtype=np.uint16, mode='r')
    val_indices = np.load(val_idx_file) # shape: (40, 32)
    max_train_idx = len(train_data) - args.block_size - 1

    # Guaranteed identical training sequence across architectures for same seed
    train_rng = np.random.RandomState(args.seed)

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

    # Schedule
    tokens_per_step   = args.batch_size * args.block_size
    schedule_steps    = math.ceil(args.total_schedule_tokens / tokens_per_step)
    warmup_steps      = max(1, int(schedule_steps * 0.02))
    print(f"[{m_label}] Schedule horizon = {schedule_steps} steps ({args.total_schedule_tokens/1e6:.1f}M tok), warmup={warmup_steps}")

    milestone_steps = {}
    for mtok in STAGE6_MILESTONE_TOKENS:
        m_step = math.ceil(mtok / tokens_per_step)
        if m_step <= schedule_steps:
            milestone_steps[m_step] = mtok

    run_id    = f"wt103_{args.model.lower()}_seed{args.seed}"
    log_jsonl = os.path.join(args.output_dir, f"{run_id}.jsonl")
    log_csv   = os.path.join(args.output_dir, f"{run_id}.csv")
    best_ckpt = os.path.join(args.ckpt_dir, f"{run_id}_best.pt")
    final_ckpt = os.path.join(args.ckpt_dir, f"{run_id}_50M.pt")
    milestone_json = os.path.join(args.output_dir, f"{run_id}_milestones.json")

    if os.path.exists(log_csv) or os.path.exists(final_ckpt):
        raise RuntimeError(f"ABORT: Output target already exists! log_csv={log_csv}, final_ckpt={final_ckpt}. Overwriting is strictly forbidden.")

    history = []
    milestone_results = []
    best_val_loss = float('inf')

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Step 0 initial validation
    val_loss_init, val_ppl_init = evaluate_deterministic_val(
        model, val_data, val_indices, block_size=args.block_size, device=device)
    print(f"Step 0 | Initial Val Loss={val_loss_init:.4f} PPL={val_ppl_init:.2f}")

    model.train()
    t_start = time.perf_counter()
    last_temp_check_time = time.time()

    for step in range(schedule_steps):
        lr = get_lr(step, schedule_steps, warmup_steps, args.peak_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Sample batch deterministically
        ix = train_rng.randint(0, max_train_idx, size=args.batch_size)
        x_list = [train_data[i : i + args.block_size].astype(np.int64) for i in ix]
        y_list = [train_data[i + 1 : i + 1 + args.block_size].astype(np.int64) for i in ix]
        x = torch.from_numpy(np.stack(x_list))
        y = torch.from_numpy(np.stack(y_list))
        if 'cuda' in str(device) and torch.cuda.is_available():
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)

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
        is_last_step = (current_step == schedule_steps)
        is_milestone = current_step in milestone_steps
        do_eval = (current_step % args.eval_interval == 0) or is_last_step or is_milestone

        # Temperature check
        now = time.time()
        if now - last_temp_check_time > 60.0:
            gpu_temp, gpu_util, _ = get_gpu_health()
            last_temp_check_time = now
            if gpu_temp > 87.0:
                print(f"\n[ALERT] GPU Temp={gpu_temp}°C (>87°C threshold)! Cooling for 30s...")
                time.sleep(30.0)

        if do_eval:
            val_loss, val_ppl = evaluate_deterministic_val(
                model, val_data, val_indices, block_size=args.block_size, device=device)
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

            print(f"Step {current_step:5d}/{schedule_steps} ({tokens_done/1e6:5.1f}M tok) | "
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
                milestone_results.append(milestone_entry)
                
                # Checkpoint at 50M
                if abs(mtok - 50_000_000) < 100_000:
                    m_ckpt = os.path.join(args.ckpt_dir, f"{run_id}_50M.pt")
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
                    print(f"  >>> SAVED FINAL 50M CHECKPOINT: {m_ckpt}")
                else:
                    print(f"  >>> MILESTONE {mtok/1e6:.0f}M | PPL={val_ppl:.2f}")

    # Save CSV
    df = pd.DataFrame(history)
    df.to_csv(log_csv, index=False)

    # Save milestone summary JSON
    with open(milestone_json, "w") as f:
        json.dump({
            "model": args.model,
            "seed": args.seed,
            "dataset": "WikiText-103",
            "non_emb_params": non_emb_params,
            "best_val_loss": best_val_loss,
            "best_val_ppl": math.exp(best_val_loss) if best_val_loss < 20 else float('inf'),
            "milestones": sorted(milestone_results, key=lambda x: x['tokens_M']),
        }, f, indent=2)

    duration = time.perf_counter() - t_start
    print(f"[{m_label}] Done in {duration:.1f}s | Best Val Loss={best_val_loss:.4f} | Logs: {log_csv}")
    return log_csv, best_ckpt, milestone_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6 Training on WikiText-103")
    parser.add_argument("--model", type=str, required=True, choices=["seq24", "par24_2", "par24_3", "seq8_wide"])
    parser.add_argument("--seed",                  type=int,   default=42)
    parser.add_argument("--total_schedule_tokens", type=int,   default=50_000_000)
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
    parser.add_argument("--data_dir",              type=str,   default="d:/data/wikitext103_gpt2")
    parser.add_argument("--output_dir",            type=str,   default="logs_stage6")
    parser.add_argument("--ckpt_dir",              type=str,   default="checkpoints_stage6")
    args = parser.parse_args()
    train(args)
