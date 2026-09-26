"""
branch_analysis_stage6.py - Diagnostic analysis for PAR24_3 on WikiText-103:
1. Branch ablation evaluation (Section 17) -> results_stage6/par24_3_branch_ablation.csv
2. Representation differentiation (pairwise cosine & update norms across 8 stages) (Section 18) -> results_stage6/par24_3_representation.json
3. Official test set evaluation (Section 24) on d:/data/wikitext103_gpt2/test.bin
"""

import os
import json
import math
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from model import GPT, GPTConfig

# ── 1. Branch ablation on PAR24_3 ─────────────────────────────────────────────
@torch.no_grad()
def evaluate_branch_ablation(ckpt_path="checkpoints_stage6/wt103_par24_3_seed42_best.pt", data_dir="d:/data/wikitext103_gpt2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_bin = os.path.join(data_dir, "val.bin")
    val_indices = np.load(os.path.join(data_dir, "validation_indices.npy"))
    val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = GPT(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    modes = ["full", "branch1", "branch2", "branch3", "random_single", "random_two"]
    ablation_results = []
    num_batches, batch_size = val_indices.shape
    block_size = config.block_size
    
    print("\n" + "="*70)
    print("STAGE 6B BRANCH ABLATION DIAGNOSTIC (PAR24_3, 8 STAGES x 3 BRANCHES)")
    print("="*70)
    
    for mode in modes:
        losses = []
        for b_idx in range(num_batches):
            batch_offsets = val_indices[b_idx]
            x_list = [val_data[i : i + block_size].astype(np.int64) for i in batch_offsets]
            y_list = [val_data[i + 1 : i + 1 + block_size].astype(np.int64) for i in batch_offsets]
            x = torch.from_numpy(np.stack(x_list)).to(device)
            y = torch.from_numpy(np.stack(y_list)).to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, targets=y, ablation_mode=mode)
            losses.append(loss.item())
        mean_loss = float(np.mean(losses))
        ppl = float(math.exp(mean_loss)) if mean_loss < 20 else float('inf')
        
        ablation_results.append({
            "ablation_mode": mode,
            "val_loss": round(mean_loss, 4),
            "val_perplexity": round(ppl, 2)
        })
        print(f"Mode: {mode:15s} | Val Loss: {mean_loss:.4f} | Val PPL: {ppl:.2f}")
        
    out_csv = "results_stage6/par24_3_branch_ablation.csv"
    pd.DataFrame(ablation_results).to_csv(out_csv, index=False)
    print(f"Saved branch ablation to {out_csv}")
    del model
    torch.cuda.empty_cache()
    return ablation_results

# ── 2. Representation differentiation (8 stages x 3 branches) ─────────────────
@torch.no_grad()
def evaluate_representation_differentiation(ckpt_path="checkpoints_stage6/wt103_par24_3_seed42_best.pt", data_dir="d:/data/wikitext103_gpt2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_bin = os.path.join(data_dir, "val.bin")
    val_indices = np.load(os.path.join(data_dir, "validation_indices.npy"))
    val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = GPT(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    num_stages = 8
    stage_cos_12 = [[] for _ in range(num_stages)]
    stage_cos_13 = [[] for _ in range(num_stages)]
    stage_cos_23 = [[] for _ in range(num_stages)]
    stage_norm1  = [[] for _ in range(num_stages)]
    stage_norm2  = [[] for _ in range(num_stages)]
    stage_norm3  = [[] for _ in range(num_stages)]
    
    num_batches = min(20, len(val_indices))
    block_size = config.block_size
    
    print("\n" + "="*70)
    print("STAGE 6B REPRESENTATION DIFFERENTIATION (PAR24_3, 8 STAGES x 3 BRANCHES)")
    print("="*70)
    
    for b_idx in range(num_batches):
        batch_offsets = val_indices[b_idx]
        x_list = [val_data[i : i + block_size].astype(np.int64) for i in batch_offsets]
        x = torch.from_numpy(np.stack(x_list)).to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, _, stage_deltas, _ = model(x, return_representations=True)
            
        for s_idx in range(num_stages):
            deltas = stage_deltas[s_idx]
            d1 = deltas[0].float()
            d2 = deltas[1].float()
            d3 = deltas[2].float()
            
            c12 = F.cosine_similarity(d1, d2, dim=-1).mean().item()
            c13 = F.cosine_similarity(d1, d3, dim=-1).mean().item()
            c23 = F.cosine_similarity(d2, d3, dim=-1).mean().item()
            
            stage_cos_12[s_idx].append(c12)
            stage_cos_13[s_idx].append(c13)
            stage_cos_23[s_idx].append(c23)
            stage_norm1[s_idx].append(d1.norm(dim=-1).mean().item())
            stage_norm2[s_idx].append(d2.norm(dim=-1).mean().item())
            stage_norm3[s_idx].append(d3.norm(dim=-1).mean().item())
            
    results = {
        "model": "par24_3",
        "num_stages": num_stages,
        "stages": []
    }
    
    print(f"Stage | cos(B1,B2) | cos(B1,B3) | cos(B2,B3) | Mean Pairwise Cos | Norm B1 | Norm B2 | Norm B3")
    print(f"-----------------------------------------------------------------------------------------")
    for s in range(num_stages):
        m12 = float(np.mean(stage_cos_12[s]))
        m13 = float(np.mean(stage_cos_13[s]))
        m23 = float(np.mean(stage_cos_23[s]))
        mean_pairwise = (m12 + m13 + m23) / 3.0
        min_cos = min(m12, m13, m23)
        max_cos = max(m12, m13, m23)
        n1 = float(np.mean(stage_norm1[s]))
        n2 = float(np.mean(stage_norm2[s]))
        n3 = float(np.mean(stage_norm3[s]))
        
        info = {
            "stage": s + 1,
            "cos_12": round(m12, 4),
            "cos_13": round(m13, 4),
            "cos_23": round(m23, 4),
            "mean_pairwise_cosine": round(mean_pairwise, 4),
            "min_cosine": round(min_cos, 4),
            "max_cosine": round(max_cos, 4),
            "norm_b1": round(n1, 4),
            "norm_b2": round(n2, 4),
            "norm_b3": round(n3, 4),
        }
        results["stages"].append(info)
        print(f"  {s+1:2d}  |   {m12:6.4f}   |   {m13:6.4f}   |   {m23:6.4f}   |      {mean_pairwise:7.4f}      | {n1:7.4f} | {n2:7.4f} | {n3:7.4f}")
        
    out_json = "results_stage6/par24_3_representation.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved representation differentiation to {out_json}")
    del model
    torch.cuda.empty_cache()
    return results

# ── 3. Official Test-Set Evaluation ───────────────────────────────────────────
@torch.no_grad()
def evaluate_official_test_set(data_dir="d:/data/wikitext103_gpt2", block_size=256, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_bin = os.path.join(data_dir, "test.bin")
    test_data = np.memmap(test_bin, dtype=np.uint16, mode='r')
    num_tokens = len(test_data)
    
    # Non-overlapping sequential slices of length block_size
    num_sequences = (num_tokens - 1) // block_size
    print(f"\nEvaluating Official WikiText-103 Test Set ({num_tokens:,} tokens, {num_sequences} sequences)...")
    
    models = {
        "SEQ24": "checkpoints_stage6/wt103_seq24_seed42_best.pt",
        "PAR24_2": "checkpoints_stage6/wt103_par24_2_seed42_best.pt",
        "PAR24_3": "checkpoints_stage6/wt103_par24_3_seed42_best.pt",
    }
    if os.path.exists("checkpoints_stage6/wt103_seq8_wide_seed42_best.pt"):
        models["SEQ8_WIDE"] = "checkpoints_stage6/wt103_seq8_wide_seed42_best.pt"
        
    test_results = {}
    
    for name, ckpt_path in models.items():
        if not os.path.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt['config']
        model = GPT(config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        losses = []
        # Process in batches
        for start_seq in range(0, num_sequences, batch_size):
            end_seq = min(start_seq + batch_size, num_sequences)
            b_size = end_seq - start_seq
            x_list = []
            y_list = []
            for s in range(start_seq, end_seq):
                offset = s * block_size
                x_list.append(test_data[offset : offset + block_size].astype(np.int64))
                y_list.append(test_data[offset + 1 : offset + 1 + block_size].astype(np.int64))
            x = torch.from_numpy(np.stack(x_list)).to(device)
            y = torch.from_numpy(np.stack(y_list)).to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
            losses.append(loss.item() * b_size)
            
        total_loss = sum(losses) / num_sequences
        ppl = math.exp(total_loss)
        test_results[name] = {
            "test_loss": round(total_loss, 4),
            "test_ppl": round(ppl, 2)
        }
        print(f"Official Test Set | {name:10s} -> Loss: {total_loss:.4f} | PPL: {ppl:.2f}")
        del model
        torch.cuda.empty_cache()
        
    out_json = "results_stage6/test_set_results.json"
    with open(out_json, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Saved official test results to {out_json}")
    return test_results

if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "all"
    if action in ("all", "ablation"):
        evaluate_branch_ablation()
    if action in ("all", "rep"):
        evaluate_representation_differentiation()
    if action in ("all", "test"):
        evaluate_official_test_set()
