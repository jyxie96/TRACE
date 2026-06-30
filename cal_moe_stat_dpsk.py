"""
Compute MoE routing statistics for DeepSeek MoE models.
Uses MoEGate directly (handles group_limited_greedy routing internally).

Usage:
    MOE_MODEL_PATH=/path/to/deepseek-moe python moe_stat_deepseek.py
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DATASET_CONFIGS = [
    ("bio_forget", "wmdp-bio-forget-corpus", None),
    ("bio_retain", "wmdp-corpora", "bio-retain-corpus"),
    ("wiki", "/data/xiejingyi/dataset/wikitext", "wikitext-2-raw-v1"),
    ("cyber_forget", "wmdp-corpora", "cyber-forget-corpus"),
    ("cyber_retain", "wmdp-corpora", "cyber-retain-corpus"),
]


def load_corpus(
    name: str,
    subset: Optional[str],
    split: str = "train",
    num_samples: int = 25,
    cache_dir: str = "./.cache",
    min_text_length: int = 1000,
) -> List[str]:
    if subset:
        ds = load_dataset(name, subset, cache_dir=cache_dir, trust_remote_code=True)[split]
    else:
        ds = load_dataset(name, cache_dir=cache_dir, trust_remote_code=True)[split]
    texts = []
    for ex in ds:
        if len(texts) >= num_samples:
            break
        t = ex.get("text") or ""
        if len(t) >= min_text_length:
            texts.append(t)
    return texts


class MoEStat:
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gating_stats: Dict[int, Dict[str, List[float]]] = {}
        self.gating_count: Dict[int, Dict[str, List[float]]] = {}
        self.token_counts: Dict[int, List[int]] = {}

    def _find_moe_layers(self) -> Dict[int, int]:
        """Find MoE layers and their number of routed experts.
        Returns {layer_idx: n_routed_experts}."""
        result = {}
        for layer_idx, layer in enumerate(self.model.model.layers):
            moe = getattr(layer, 'mlp', None)
            if moe is not None and hasattr(moe, 'gate') and hasattr(moe, 'experts'):
                n_experts = moe.gate.n_routed_experts
                result[layer_idx] = n_experts
        return result

    def extract_moe_activations(self, inputs: Dict, layer_idx: int) -> Optional[torch.Tensor]:
        """Forward model, capture pre-MoE hidden states at given layer via hook."""
        stored = [None]

        def hook_fn(module, input):
            stored[0] = input[0].detach()

        moe = self.model.model.layers[layer_idx].mlp
        handle = moe.register_forward_pre_hook(hook_fn)
        with torch.no_grad():
            self.model(**inputs)
        handle.remove()
        return stored[0]  # (bsz, seq_len, hidden_dim)

    def get_gating_counts(self, pre_hidden: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call MoEGate on captured hidden states to get routing decisions.
        
        DeepSeek MoEGate.forward(hidden_states_3d) returns:
            topk_idx: (bsz*seq_len, top_k) 
            topk_weight: (bsz*seq_len, top_k)
            aux_loss: scalar or None
        
        Returns:
            gating_scores: (bsz, seq_len, n_experts) - routing weights per expert
            gating_count: (bsz, seq_len, n_experts) - binary selection count per expert
        """
        moe = self.model.model.layers[layer_idx].mlp
        gate = moe.gate
        n_experts = gate.n_routed_experts
        bsz, seq_len, _ = pre_hidden.shape

        with torch.no_grad():
            topk_idx, topk_weight, _ = gate(pre_hidden)  # (N, top_k), (N, top_k)

        N = bsz * seq_len
        top_k = topk_idx.shape[1]

        # Build per-expert gating scores and counts
        gating_scores = torch.zeros(N, n_experts, dtype=torch.float32, device=pre_hidden.device)
        gating_count = torch.zeros(N, n_experts, dtype=torch.float32, device=pre_hidden.device)

        for k in range(top_k):
            gating_scores.scatter_(1, topk_idx[:, k:k+1], topk_weight[:, k:k+1])
            gating_count.scatter_(1, topk_idx[:, k:k+1], torch.ones_like(topk_weight[:, k:k+1]))

        gating_scores = gating_scores.view(bsz, seq_len, n_experts)
        gating_count = gating_count.view(bsz, seq_len, n_experts)
        return gating_scores, gating_count

    def initialize_stats(self, moe_layers: Dict[int, int]):
        for layer_idx, n_experts in moe_layers.items():
            self.gating_stats[layer_idx] = {f"expert_{i}": [] for i in range(n_experts)}
            self.gating_count[layer_idx] = {f"expert_{i}": [] for i in range(n_experts)}
            self.token_counts[layer_idx] = []

    def collect_batch(self, layer_idx: int, gating_scores: torch.Tensor, gating_count: torch.Tensor):
        bsz, seq_len, n_experts = gating_scores.shape
        self.token_counts[layer_idx].append(bsz * seq_len)
        for e in range(n_experts):
            key = f"expert_{e}"
            self.gating_stats[layer_idx][key].append(gating_scores[:, :, e].sum().item())
            self.gating_count[layer_idx][key].append(gating_count[:, :, e].sum().item())

    def compute_statistics(self, calibration_data: List[str], max_length: int = 2000) -> None:
        moe_layers = self._find_moe_layers()
        self.initialize_stats(moe_layers)

        for layer_idx in moe_layers:
            print(f"  Processing layer {layer_idx} ({moe_layers[layer_idx]} experts)...")
            for i, text in enumerate(calibration_data):
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                for k in ("input_ids", "attention_mask", "token_type_ids"):
                    if k in inputs:
                        inputs[k] = inputs[k].long()
                if inputs["input_ids"].numel() == 0:
                    continue

                pre_hidden = self.extract_moe_activations(inputs, layer_idx)
                if pre_hidden is None:
                    continue

                gating_scores, gating_count = self.get_gating_counts(pre_hidden, layer_idx)
                self.collect_batch(layer_idx, gating_scores, gating_count)

    def get_averaged_stats(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        layers = sorted(self.gating_stats.keys())
        if not layers:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        max_experts = max(len(self.gating_stats[l]) for l in layers)
        total_tokens_per_layer = {l: sum(self.token_counts.get(l, [])) for l in layers}

        gating_df = pd.DataFrame(index=[f"expert_{i}" for i in range(max_experts)])
        count_df = pd.DataFrame(index=[f"expert_{i}" for i in range(max_experts)])
        freq_df = pd.DataFrame(index=[f"expert_{i}" for i in range(max_experts)])

        for l in layers:
            tot = total_tokens_per_layer[l]
            gating_df[f"layer_{l}"] = [
                np.sum(self.gating_stats[l][f"expert_{e}"]) / tot if tot else np.nan
                for e in range(max_experts)
            ]
            count_sum_e = [np.sum(self.gating_count[l][f"expert_{e}"]) for e in range(max_experts)]
            count_df[f"layer_{l}"] = count_sum_e
            freq_df[f"layer_{l}"] = [
                count_sum_e[e] / tot if tot else np.nan
                for e in range(max_experts)
            ]
        return gating_df, count_df, freq_df

    def save_to_csv(self, output_path: str) -> None:
        gating_df, count_df, freq_df = self.get_averaged_stats()
        base = output_path.rsplit(".", 1)[0]
        gating_df.to_csv(f"{base}_gating.csv")
        count_df.to_csv(f"{base}_gating_count.csv")
        freq_df.to_csv(f"{base}_expert_frequency.csv")
        print(f"  Saved to {base}_*.csv")


def run_one_corpus(
    name: str,
    subset: Optional[str],
    model,
    tokenizer,
    device: str,
    num_samples: int,
    max_length: int,
    out_path: str,
    cache_dir: str,
) -> None:
    texts = load_corpus(name, subset, num_samples=num_samples, cache_dir=cache_dir)
    if not texts:
        print(f"  No texts loaded, skipping")
        return
    print(f"  Loaded {len(texts)} texts")
    stat = MoEStat(model, tokenizer, device)
    stat.compute_statistics(texts, max_length=max_length)
    stat.save_to_csv(out_path)


def main():
    model_path = os.environ.get("MOE_MODEL_PATH", "/data/xiejingyi/WAGLE/files/results/unlearn_wmdp_cyber/GDSelected/2026-03-22-21-49-17-759366/checkpoints")
    out_dir = os.environ.get("MOE_STAT_OUTPUT", f"/data/xiejingyi/WAGLE/plot/GD-Selected-gating-{os.path.basename(model_path)}-statistics")
    cache_dir = os.environ.get("HF_CACHE", "./.cache")
    num_samples = int(os.environ.get("MOE_NUM_SAMPLES", "25"))
    max_length = int(os.environ.get("MOE_MAX_LENGTH", "1024"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    device = next(model.parameters()).device

    for label, name, subset in DATASET_CONFIGS:
        print(f"\nProcessing: {label}")
        out_path = os.path.join(out_dir, f"{label}.csv")
        run_one_corpus(name, subset, model, tokenizer, str(device), num_samples, max_length, out_path, cache_dir)


if __name__ == "__main__":
    main()