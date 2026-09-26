"""
data.py - Data preparation and loading for Stage-Parallel Decoder experiments.
Supports TinyShakespeare (smoke test) and TinyStories (main experiment).
Tokenizer: tiktoken GPT-2 BPE.
"""

import os
import sys
import numpy as np
import torch
import tiktoken

def get_tokenizer():
    return tiktoken.get_encoding("gpt2")

def prepare_shakespeare(data_dir="data/shakespeare"):
    os.makedirs(data_dir, exist_ok=True)
    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_bin) and os.path.exists(val_bin):
        print(f"[Shakespeare] Found cached data in {data_dir}")
        return train_bin, val_bin

    source_path = "D:/data/shakespeare_char/input.txt"
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Shakespeare text not found at {source_path}")

    with open(source_path, "r", encoding="utf-8") as f:
        text = f.read()

    enc = get_tokenizer()
    tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
    tokens = np.array(tokens, dtype=np.uint16)

    # 90% train, 10% val
    n = len(tokens)
    train_tokens = tokens[:int(n * 0.9)]
    val_tokens = tokens[int(n * 0.9):]

    train_tokens.tofile(train_bin)
    val_tokens.tofile(val_bin)
    print(f"[Shakespeare] Processed {n} tokens. Train: {len(train_tokens)}, Val: {len(val_tokens)}")
    return train_bin, val_bin

def prepare_tinystories(data_dir="D:/data/tinystories_10m", target_train_tokens=10_000_000, target_val_tokens=800_000):
    os.makedirs(data_dir, exist_ok=True)
    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")
    meta_file = os.path.join(data_dir, "meta.txt")

    if os.path.exists(train_bin) and os.path.exists(val_bin) and os.path.exists(meta_file):
        train_size = os.path.getsize(train_bin) // 2
        val_size = os.path.getsize(val_bin) // 2
        if train_size >= target_train_tokens and val_size >= target_val_tokens:
            print(f"[TinyStories] Found cached data in {data_dir}: {train_size} train tokens, {val_size} val tokens")
            return train_bin, val_bin

    print(f"[TinyStories] Streaming and preparing {target_train_tokens} train tokens and {target_val_tokens} val tokens...")
    import datasets
    enc = get_tokenizer()
    eot = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    # Stream train
    print("Loading train stream...")
    ds_train = datasets.load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    train_tokens = []
    for example in ds_train:
        t = enc.encode_ordinary(example["text"])
        t.append(eot)
        train_tokens.extend(t)
        if len(train_tokens) >= target_train_tokens:
            break

    train_arr = np.array(train_tokens[:target_train_tokens], dtype=np.uint16)
    train_arr.tofile(train_bin)
    print(f"[TinyStories] Saved {len(train_arr)} train tokens to {train_bin}")

    # Stream validation
    print("Loading validation stream...")
    ds_val = datasets.load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    val_tokens = []
    for example in ds_val:
        t = enc.encode_ordinary(example["text"])
        t.append(eot)
        val_tokens.extend(t)
        if len(val_tokens) >= target_val_tokens:
            break

    val_arr = np.array(val_tokens[:target_val_tokens], dtype=np.uint16)
    val_arr.tofile(val_bin)
    print(f"[TinyStories] Saved {len(val_arr)} val tokens to {val_bin}")

    with open(meta_file, "w", encoding="utf-8") as f:
        f.write(f"train_tokens={len(train_arr)}\nval_tokens={len(val_arr)}\ntokenizer=gpt2\n")

    return train_bin, val_bin

class TokenDataset:
    def __init__(self, data_file, block_size):
        self.data_file = data_file
        self.block_size = block_size
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)

    def get_batch(self, batch_size, device='cuda'):
        max_idx = self.total_tokens - self.block_size - 1
        ix = np.random.randint(0, max_idx, size=batch_size)
        x_list = [self.data[i : i + self.block_size].astype(np.int64) for i in ix]
        y_list = [self.data[i + 1 : i + 1 + self.block_size].astype(np.int64) for i in ix]
        x = torch.from_numpy(np.stack(x_list))
        y = torch.from_numpy(np.stack(y_list))
        if 'cuda' in str(device) and torch.cuda.is_available():
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y

    def get_sequential_batch(self, start_idx, batch_size, device='cuda'):
        x_list = []
        y_list = []
        for b in range(batch_size):
            offset = start_idx + b * self.block_size
            if offset + self.block_size + 1 > self.total_tokens:
                offset = offset % (self.total_tokens - self.block_size - 1)
            x_list.append(self.data[offset : offset + self.block_size].astype(np.int64))
            y_list.append(self.data[offset + 1 : offset + 1 + self.block_size].astype(np.int64))
        x = torch.from_numpy(np.stack(x_list)).to(device)
        y = torch.from_numpy(np.stack(y_list)).to(device)
        return x, y

if __name__ == "__main__":
    print("Testing data preparation...")
    s_tr, s_va = prepare_shakespeare("data/shakespeare")
    print(f"Shakespeare ready: {s_tr}, {s_va}")
