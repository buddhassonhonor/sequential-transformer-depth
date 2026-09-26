"""
prepare_wikitext103.py - Downloads and tokenizes official Salesforce/wikitext (wikitext-103-raw-v1).
Stores the tokenized dataset in a configurable output directory as uint16 memmap binary files:
- train.bin
- val.bin
- test.bin
- meta.json
- validation_indices.npy

Follows Stage 6 specifications strictly:
- GPT-2 BPE tokenizer (tiktoken)
- Inserts <|endoftext|> between articles
- Generates deterministic validation batch indices
"""

import argparse
import os
import re
import json
import time
import numpy as np
import datasets
import tiktoken

def main():
    parser = argparse.ArgumentParser(description="Prepare WikiText-103 with GPT-2 BPE tokenization.")
    parser.add_argument("--output-dir", default="data/wikitext103_gpt2")
    parser.add_argument("--results-dir", default="results_stage6")
    args = parser.parse_args()

    out_dir = args.output_dir
    results_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    h1_pattern = re.compile(r'^\s*=\s+[^=]+?\s+=\s*$')

    splits = ["train", "validation", "test"]
    split_files = {
        "train": os.path.join(out_dir, "train.bin"),
        "validation": os.path.join(out_dir, "val.bin"),
        "test": os.path.join(out_dir, "test.bin"),
    }
    
    stats = {}
    
    print(f"Loading official Salesforce/wikitext wikitext-103-raw-v1...")
    dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    
    for split_name in splits:
        out_path = split_files[split_name]
        ds_split = dataset[split_name]
        num_rows = len(ds_split)
        print(f"\nProcessing '{split_name}' split ({num_rows:,} rows)...")
        
        t0 = time.time()
        all_tokens = []
        num_articles = 0
        
        for i, row in enumerate(ds_split):
            text = row["text"]
            if not text:
                continue
            
            # Check for article boundary (H1 header)
            if h1_pattern.match(text):
                num_articles += 1
                if len(all_tokens) > 0:
                    all_tokens.append(eot_token)
            
            toks = enc.encode_ordinary(text)
            all_tokens.extend(toks)
            
            if (i + 1) % 200_000 == 0:
                print(f"  Row {i+1:,}/{num_rows:,}: {len(all_tokens):,} tokens accumulated...")
        
        # Final trailing eot
        if len(all_tokens) > 0 and all_tokens[-1] != eot_token:
            all_tokens.append(eot_token)
            
        token_arr = np.array(all_tokens, dtype=np.uint16)
        token_arr.tofile(out_path)
        dur = time.time() - t0
        
        file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"Finished '{split_name}': {len(token_arr):,} tokens, {num_articles:,} articles.")
        print(f"Saved to {out_path} ({file_size_mb:.2f} MB) in {dur:.1f}s.")
        
        stats[split_name] = {
            "num_tokens": int(len(token_arr)),
            "num_articles": int(num_articles),
            "file_size_bytes": int(os.path.getsize(out_path)),
            "file_name": os.path.basename(out_path)
        }
    
    # Generate deterministic validation positions
    # 40 validation batches of batch_size=32, block_size=256 -> 327,680 tokens
    val_tokens_total = stats["validation"]["num_tokens"]
    block_size = 256
    num_val_batches = 40
    batch_size = 32
    max_val_idx = val_tokens_total - block_size - 1
    
    # Deterministic validation indices generator
    rng = np.random.RandomState(42)
    val_indices = rng.randint(0, max_val_idx, size=(num_val_batches, batch_size))
    val_idx_path = os.path.join(out_dir, "validation_indices.npy")
    np.save(val_idx_path, val_indices)
    print(f"\nSaved deterministic validation batch indices to {val_idx_path}")
    
    # Save copy to results_stage6/
    np.save(os.path.join(results_dir, "validation_indices.npy"), val_indices)
    
    meta = {
        "dataset": "wikitext-103-raw-v1",
        "repository": "Salesforce/wikitext",
        "tokenizer": "gpt2 (tiktoken)",
        "vocab_size": 50304,
        "token_dtype": "uint16",
        "eot_token_id": eot_token,
        "splits": stats,
        "deterministic_val_batches": num_val_batches,
        "val_batch_size": batch_size,
        "block_size": block_size
    }
    
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")
    
    stats_path = os.path.join(results_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved dataset stats to {stats_path}")

if __name__ == "__main__":
    main()
