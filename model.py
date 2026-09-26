"""
model.py - Stage-Parallel Decoder Transformer implementation.
NanoGPT-style clean architecture supporting:
- SEQ3: 3 sequential Transformer blocks
- SEQ6: 6 sequential Transformer blocks (primary baseline)
- SEQ6_WIDE: 6 sequential blocks with wide FFN (parameter-matched to SEQ12/PAR12_2)
- PAR6_MEAN: 3 stages x 2 parallel branches with sqrt(2) mean residual aggregation
- PAR6_GATE: 3 stages x 2 parallel branches with learnable per-stage softmax gate
- SEQ12: 12 sequential Transformer blocks (Stage 2 quality upper baseline)
- PAR12_2: 6 stages x 2 parallel branches (Stage 2 core model, critical depth = 6)
- PAR12_3: 4 stages x 3 parallel branches (Stage 2 scaling model, critical depth = 4)
Supports branch ablation evaluation ('full', 'branch1', 'branch2', 'random_single').
"""

import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50304  # GPT-2 vocab 50257 padded to multiple of 64
    n_layer: int = 6         # total number of blocks
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 0            # FFN hidden dim; 0 means use 4*d_model (standard)
    dropout: float = 0.1
    model_type: str = "seq6" # 'seq3', 'seq6', 'seq6_wide', 'par6_mean', 'par6_gate', 'seq12', 'par12_2', 'par12_3', 'seq24', 'par24_2', 'par24_3', 'seq8_wide'
    weight_tying: bool = True

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # FlashAttention / Memory efficient SDPA with causal mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        d_ff = config.d_ff if config.d_ff > 0 else 4 * config.d_model
        self.c_fc = nn.Linear(config.d_model, d_ff, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ParallelStage(nn.Module):
    def __init__(self, config: GPTConfig, num_branches: int = 2, aggregation: str = "mean"):
        super().__init__()
        self.num_branches = num_branches
        self.aggregation = aggregation # 'mean' or 'gate'
        self.branches = nn.ModuleList([TransformerBlock(config) for _ in range(num_branches)])
        if self.aggregation == "gate":
            self.gate_logits = nn.Parameter(torch.zeros(num_branches))
        else:
            self.gate_logits = None

    def get_gate_weights(self):
        if self.aggregation == "gate":
            return F.softmax(self.gate_logits, dim=0).detach()
        else:
            return torch.full((self.num_branches,), 1.0 / self.num_branches, device=next(self.parameters()).device)

    def forward(self, x, return_deltas: bool = False, ablation_mode: str = "full"):
        # Memory-safe incremental summation for large K (Rule 8)
        if not return_deltas and ablation_mode == "full":
            if self.aggregation == "mean":
                scale = 1.0 / math.sqrt(self.num_branches)
                delta_sum = None
                for branch in self.branches:
                    y = branch(x)
                    d = y - x
                    delta_sum = d if delta_sum is None else (delta_sum + d)
                x_next = x + delta_sum * scale
                return x_next
            elif self.aggregation == "gate":
                alpha = F.softmax(self.gate_logits, dim=0)
                weighted_delta = None
                for i, branch in enumerate(self.branches):
                    y = branch(x)
                    d = alpha[i] * (y - x)
                    weighted_delta = d if weighted_delta is None else (weighted_delta + d)
                x_next = x + math.sqrt(self.num_branches) * weighted_delta
                return x_next

        # General path (used when return_deltas=True or in ablation modes)
        branch_deltas = []
        for branch in self.branches:
            y = branch(x)
            delta = y - x
            branch_deltas.append(delta)

        if ablation_mode == "full":
            if self.aggregation == "mean":
                scale = 1.0 / math.sqrt(self.num_branches)
                agg_delta = torch.stack(branch_deltas, dim=0).sum(dim=0) * scale
                x_next = x + agg_delta
            elif self.aggregation == "gate":
                alpha = F.softmax(self.gate_logits, dim=0)
                weighted_delta = sum(alpha[i] * branch_deltas[i] for i in range(self.num_branches))
                x_next = x + math.sqrt(self.num_branches) * weighted_delta
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")
        elif ablation_mode == "branch1":
            x_next = x + branch_deltas[0]
        elif ablation_mode == "branch2":
            x_next = x + branch_deltas[1]
        elif ablation_mode == "branch3":
            x_next = x + branch_deltas[2]
        elif ablation_mode == "random_single":
            chosen = np.random.randint(0, self.num_branches)
            x_next = x + branch_deltas[chosen]
        elif ablation_mode == "random_two":
            indices = np.random.choice(self.num_branches, size=2, replace=False)
            agg = (branch_deltas[indices[0]] + branch_deltas[indices[1]]) / math.sqrt(2)
            x_next = x + agg
        else:
            raise ValueError(f"Unknown ablation_mode: {ablation_mode}")

        if return_deltas:
            return x_next, branch_deltas
        return x_next

    def forward_concurrent(self, x, streams):
        assert len(streams) == self.num_branches, f"Require {self.num_branches} streams"
        branch_outputs = [None] * self.num_branches
        current_stream = torch.cuda.current_stream()

        for i, (branch, stream) in enumerate(zip(self.branches, streams)):
            stream.wait_stream(current_stream)
            with torch.cuda.stream(stream):
                y = branch(x)
                branch_outputs[i] = y - x

        for stream in streams:
            current_stream.wait_stream(stream)

        if self.aggregation == "mean":
            scale = 1.0 / math.sqrt(self.num_branches)
            agg_delta = torch.stack(branch_outputs, dim=0).sum(dim=0) * scale
            x_next = x + agg_delta
        elif self.aggregation == "gate":
            alpha = F.softmax(self.gate_logits, dim=0)
            weighted_delta = sum(alpha[i] * branch_outputs[i] for i in range(self.num_branches))
            x_next = x + math.sqrt(self.num_branches) * weighted_delta
        return x_next

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        m_type = config.model_type.lower()
        if m_type == "seq3":
            self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(3)])
            self.is_parallel = False
        elif m_type in ("seq6", "seq6_wide"):
            self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(6)])
            self.is_parallel = False
        elif m_type == "seq12":
            self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(12)])
            self.is_parallel = False
        elif m_type == "par6_mean":
            self.layers = nn.ModuleList([ParallelStage(config, num_branches=2, aggregation="mean") for _ in range(3)])
            self.is_parallel = True
        elif m_type == "par6_gate":
            self.layers = nn.ModuleList([ParallelStage(config, num_branches=2, aggregation="gate") for _ in range(3)])
            self.is_parallel = True
        elif m_type == "par12_2":
            self.layers = nn.ModuleList([ParallelStage(config, num_branches=2, aggregation="mean") for _ in range(6)])
            self.is_parallel = True
        elif m_type == "par12_3":
            self.layers = nn.ModuleList([ParallelStage(config, num_branches=3, aggregation="mean") for _ in range(4)])
            self.is_parallel = True
        elif m_type == "seq24":
            self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(24)])
            self.is_parallel = False
        elif m_type == "par24_2":
            self.layers = nn.ModuleList([ParallelStage(config, num_branches=2, aggregation="mean") for _ in range(12)])
            self.is_parallel = True
        elif m_type == "par24_3":
            self.layers = nn.ModuleList([ParallelStage(config, num_branches=3, aggregation="mean") for _ in range(8)])
            self.is_parallel = True
        elif m_type == "seq8_wide":
            self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(8)])
            self.is_parallel = False
        elif m_type.startswith("sp24_s") and "k" in m_type:
            parts = m_type[len("sp24_s"):].split("k")
            s = int(parts[0])
            k = int(parts[1])
            assert s * k == 24, f"Invalid stage/branch combination: S={s}, K={k}, must multiply to 24"
            self.layers = nn.ModuleList([ParallelStage(config, num_branches=k, aggregation="mean") for _ in range(s)])
            self.is_parallel = True
        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.weight_tying:
            self.wte.weight = self.lm_head.weight

        # Init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
            if not self.config.weight_tying:
                n_params -= self.wte.weight.numel()
        return n_params

    def forward(self, idx, targets=None, return_representations=False, concurrent_streams=None, ablation_mode: str = "full"):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        stage_deltas = []
        stage_gates = []

        for layer in self.layers:
            if self.is_parallel:
                if concurrent_streams is not None:
                    x = layer.forward_concurrent(x, concurrent_streams)
                elif return_representations:
                    x, deltas = layer(x, return_deltas=True, ablation_mode=ablation_mode)
                    stage_deltas.append(deltas)
                    stage_gates.append(layer.get_gate_weights())
                else:
                    x = layer(x, ablation_mode=ablation_mode)
            else:
                x = layer(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        if return_representations:
            return logits, loss, stage_deltas, stage_gates
        return logits, loss
