"""
train_stage7.py - Stage 7A Training Script: Sequential-Depth Frontier on WikiText-103.
Models supported: SP24_S6K4, SP24_S4K6, SP24_S3K8, SP24_S2K12, SP24_S1K24.
All models preserve exactly 24 Transformer blocks with 61,821,696 non-embedding parameters.

Key Principles:
- Completely frozen WikiText-103 dataset (GPT-2 BPE, 117.9M tokens).
- Exact match of non-embedding parameters: assert non_emb_params == 61821696.
- Incremental residual summation (memory-safe branch accumulation).
- Deterministic training token sampling (RNG seed=42) matching Stage 6 seed42.
- Deterministic validation evaluation via validation_indices.npy.
- Milestones recorded at 5M, 10M, 20M, 30M, 40M, 50M tokens.
- Strict anti-overwrite checks.
- Preflight verification mode (--preflight).
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

STAGE7_MILESTONE_TOKENS = [
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

def get_lr(step, total_steps, warmup_steps, peak_lr, min_lr):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)

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

def build_config(m_type: str, d_model: int, n_head: int, block_size: int, dropout: float):
    m_type = m_type.lower()
    return GPTConfig(
        block_size=block_size,
        vocab_size=50304,
        n_layer=24,
        n_head=n_head,
        d_model=d_model,
        d_ff=0,
        dropout=dropout,
        model_type=m_type,
        weight_tying=True,
    )

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
    val_indices = np.load(val_idx_file)
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

    # Mandatory Parameter Count Assertion
    assert non_emb_params == 61821696, f"Parameter mismatch! Expected 61821696, got {non_emb_params}"

    # Optimizer
    decay_params   = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() <  2]
    optimizer = torch.optim.AdamW(
        [{'params': decay_params,   'weight_decay': args.weight_decay},
         {'params': nodecay_params, 'weight_decay': 0.0}],
        lr=args.peak_lr, betas=(0.9, 0.95), eps=1e-8
    )

    # ── Preflight mode ────────────────────────────────────────────────────────
    if args.preflight:
        print(f"\n[PREFLIGHT] Testing {m_label} for 15 steps (forward + backward)...")
        model.train()
        for p_step in range(15):
            batch_offsets = train_rng.randint(0, max_train_idx, size=args.batch_size)
            x_list = [train_data[i : i + args.block_size].astype(np.int64) for i in batch_offsets]
            y_list = [train_data[i + 1 : i + 1 + args.block_size].astype(np.int64) for i in batch_offsets]
            x = torch.from_numpy(np.stack(x_list)).to(device)
            y = torch.from_numpy(np.stack(y_list)).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
            assert not torch.isnan(loss), "Preflight NaN loss!"
            assert not torch.isinf(loss), "Preflight Inf loss!"
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        temp, util, mem = get_gpu_health()
        peak_vram = torch.cuda.max_memory_reserved() / 1e6 if torch.cuda.is_available() else 0.0
        print(f"[PREFLIGHT PASSED] {m_label}: Loss={loss.item():.4f}, GradNorm={gnorm:.4f}, PeakVRAM={peak_vram:.1f}MB, Temp={temp}°C")
        assert peak_vram < 18000, f"Peak VRAM {peak_vram}MB exceeds 18GB threshold!"
        assert temp <= 85.0, f"GPU Temp {temp}°C exceeds 85°C threshold!"
        return

    # Schedule
    tokens_per_step   = args.batch_size * args.block_size
    schedule_steps    = math.ceil(args.total_schedule_tokens / tokens_per_step)
    warmup_steps      = max(1, int(schedule_steps * 0.02))
    print(f"[{m_label}] Schedule horizon = {schedule_steps} steps ({args.total_schedule_tokens/1e6:.1f}M tok), warmup={warmup_steps}")

    milestone_steps = {}
    for mtok in STAGE7_MILESTONE_TOKENS:
        m_step = math.ceil(mtok / tokens_per_step)
        if m_step <= schedule_steps:
            milestone_steps[m_step] = mtok

    run_id    = f"wt103_{args.model.lower()}_seed{args.seed}"
    log_jsonl = os.path.join(args.output_dir, f"{run_id}.jsonl")
    log_csv   = os.path.join(args.output_dir, f"{run_id}.csv")
    best_ckpt = os.path.join(args.ckpt_dir, f"{run_id}_best.pt")
    final_ckpt = os.path.join(args.ckpt_dir, f"{run_id}_50M.pt")
    milestone_json = os.path.join(args.output_dir, f"{run_id}_milestones.json")

    # Anti-overwrite assertion
    if os.path.exists(log_csv) or os.path.exists(final_ckpt):
        raise RuntimeError(f"ABORT: Output target already exists! log_csv={log_csv}, final_ckpt={final_ckpt}. Overwriting is forbidden.")

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

        batch_offsets = train_rng.randint(0, max_train_idx, size=args.batch_size)
        x_list = [train_data[i : i + args.block_size].astype(np.int64) for i in batch_offsets]
        y_list = [train_data[i + 1 : i + 1 + args.block_size].astype(np.int64) for i in batch_offsets]
        x = torch.from_numpy(np.stack(x_list))
        y = torch.from_numpy(np.stack(y_list))
        if 'cuda' in str(device) and torch.cuda.is_available():
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
            
        assert not torch.isnan(loss), f"NaN loss at step {step}"
        assert not torch.isinf(loss), f"Inf loss at step {step}"

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        current_step = step + 1
        tokens_done  = current_step * tokens_per_step
        is_milestone = current_step in milestone_steps
        is_eval_step = (current_step % args.eval_interval == 0) or (current_step == schedule_steps) or is_milestone

        # Periodic temperature check
        if time.time() - last_temp_check_time >= 60:
            gpu_temp, gpu_util, mem_used = get_gpu_health()
            last_temp_check_time = time.time()
            if gpu_temp > 85.0:
                print(f"\n[WARNING] High GPU temp: {gpu_temp}°C! Pausing for 30s to cool down...")
                time.sleep(30)

        if is_eval_step:
            val_loss, val_ppl = evaluate_deterministic_val(
                model, val_data, val_indices, block_size=args.block_size, device=device)

            elapsed = time.perf_counter() - t_start
            tokens_per_sec = tokens_done / elapsed if elapsed > 0 else 0.0
            peak_vram = torch.cuda.max_memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0.0
            gpu_temp, gpu_util, _ = get_gpu_health()

            log_entry = {
                "step":              current_step,
                "tokens_processed":  tokens_done,
                "train_loss":        round(loss.item(), 4),
                "val_loss":          round(val_loss, 4),
                "val_perplexity":    round(val_ppl, 2),
                "lr":                round(lr, 6),
                "grad_norm":         round(grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm), 4),
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
                    'tokens_M': round(mtok / 1e6, 1),
                    'step': current_step,
                    'val_loss': val_loss,
                    'val_ppl': val_ppl,
                    'tokens_per_second': round(tokens_per_sec, 1),
                    'peak_vram_mb': round(peak_vram, 1),
                    'gpu_temp_c': gpu_temp,
                    'wall_clock_seconds': round(elapsed, 1)
                }
                milestone_results.append(milestone_entry)
                
                # Checkpoint at 50M
                if abs(mtok - 50_000_000) < 100_000:
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
                    }, final_ckpt)
                    print(f"  >>> SAVED FINAL 50M CHECKPOINT: {final_ckpt}")
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
    parser = argparse.ArgumentParser(description="Stage 7A Training on WikiText-103")
    parser.add_argument("--model", type=str, required=True,
                        choices=["sp24_s6k4", "sp24_s4k6", "sp24_s3k8", "sp24_s2k12", "sp24_s1k24",
                                 "seq24", "par24_2", "par24_3"])
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
    parser.add_argument("--output_dir",            type=str,   default="results_stage7")
    parser.add_argument("--ckpt_dir",              type=str,   default="checkpoints_stage7")
    parser.add_argument("--preflight",             action="store_true", help="Run 15 steps preflight check")
    args = parser.parse_args()
    train(args)
