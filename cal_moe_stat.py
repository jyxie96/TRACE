import os
import torch
import torch.nn.functional as F
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
    ("cyber_retain", "wmdp-corpora", "cyber-retain-corpus")
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


def compute_expert_importance_up_proj(pre_moe_hidden: torch.Tensor, moe_layer) -> torch.Tensor:
    batch, seq_len, hidden_dim = pre_moe_hidden.shape
    N = batch * seq_len
    flat = pre_moe_hidden.reshape(N, hidden_dim)
    num_experts = moe_layer.num_experts
    norms_list = []
    for e in range(num_experts):
        proj = F.linear(flat, moe_layer.experts[e].up_proj.weight)
        norms_list.append(proj.norm(dim=-1))
    out = torch.stack(norms_list, dim=-1).view(batch, seq_len, num_experts)
    return out


class MoEStat:
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gating_stats: Dict[int, Dict[str, List[float]]] = {}
        self.expert_importance_stats: Dict[int, Dict[str, List[float]]] = {}
        self.gating_count: Dict[int, Dict[str, List[float]]] = {}
        self.token_counts: Dict[int, List[int]] = {}

    def extract_moe_activations(self, inputs: Dict, layer_idx: int) -> Optional[Dict]:
        activations = {}

        def hook_fn(module, input, output):
            if not hasattr(module, "gate"):
                return
            activations["pre_moe_hidden"] = input[0]
            if isinstance(output, tuple):
                activations["output_hidden"] = output[0]
                activations["router_logits"] = output[1]
            else:
                activations["output_hidden"] = output

        mlp = self.model.model.layers[layer_idx].mlp
        handle = mlp.register_forward_hook(hook_fn)
        with torch.no_grad():
            self.model(**inputs, output_router_logits=True)
        handle.remove()
        return activations if activations else None

    def get_gating_and_importance(self, activations: Dict, layer_idx: int) -> Optional[Tuple]:
        moe_layer = self.model.model.layers[layer_idx].mlp
        if not hasattr(moe_layer, "gate"):
            return None
        pre = activations["pre_moe_hidden"]
        batch, seq_len, hidden_dim = pre.shape
        flat = pre.view(-1, hidden_dim)
        router_logits = moe_layer.gate(flat)
        routing_weights = F.softmax(router_logits.float(), dim=1)
        top_k = moe_layer.top_k
        routing_weights, selected = torch.topk(routing_weights, top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(pre.dtype)
        num_experts = moe_layer.num_experts
        gating_scores = torch.zeros(flat.shape[0], num_experts, dtype=pre.dtype, device=pre.device)
        gating_count = torch.zeros(flat.shape[0], num_experts, dtype=pre.dtype, device=pre.device)
        for k in range(top_k):
            gating_scores.scatter_(1, selected[:, k : k + 1], routing_weights[:, k : k + 1])
            gating_count.scatter_(1, selected[:, k : k + 1], torch.ones_like(routing_weights[:, k : k + 1]))
        expert_importance = compute_expert_importance_up_proj(pre, moe_layer)
        gating_scores = gating_scores.view(batch, seq_len, num_experts)
        gating_count = gating_count.view(batch, seq_len, num_experts)
        expert_importance = expert_importance * gating_count
        return gating_scores, gating_count, expert_importance

    def collect_batch(self, layer_idx: int, gating_scores: torch.Tensor, gating_count: torch.Tensor, expert_importance: torch.Tensor):
        batch, seq_len, num_experts = gating_scores.shape
        self.token_counts[layer_idx].append(batch * seq_len)
        for e in range(num_experts):
            key = f"expert_{e}"
            self.gating_stats[layer_idx][key].append(gating_scores[:, :, e].sum().item())
            self.gating_count[layer_idx][key].append(gating_count[:, :, e].sum().item())
            self.expert_importance_stats[layer_idx][key].append(expert_importance[:, :, e].sum().item())

    def initialize_stats(self, num_layers: int, num_experts_per_layer: Dict[int, int]):
        for layer_idx in range(num_layers):
            if layer_idx not in num_experts_per_layer:
                continue
            n = num_experts_per_layer[layer_idx]
            self.gating_stats[layer_idx] = {f"expert_{i}": [] for i in range(n)}
            self.expert_importance_stats[layer_idx] = {f"expert_{i}": [] for i in range(n)}
            self.gating_count[layer_idx] = {f"expert_{i}": [] for i in range(n)}
            self.token_counts[layer_idx] = []

    def compute_statistics(self, calibration_data: List[str], max_length: int = 2000) -> None:
        num_layers = len(self.model.model.layers)
        num_experts_per_layer = {}
        for layer_idx in range(num_layers):
            mlp = self.model.model.layers[layer_idx].mlp
            if hasattr(mlp, "gate"):
                num_experts_per_layer[layer_idx] = mlp.num_experts
        self.initialize_stats(num_layers, num_experts_per_layer)

        for layer_idx in list(num_experts_per_layer.keys()):
            for demo_text in calibration_data:
                inputs = self.tokenizer(
                    demo_text,
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
                activations = self.extract_moe_activations(inputs, layer_idx)
                if not activations or "pre_moe_hidden" not in activations:
                    continue
                out = self.get_gating_and_importance(activations, layer_idx)
                if out is None:
                    continue
                gating_scores, gating_count, expert_importance = out
                self.collect_batch(layer_idx, gating_scores, gating_count, expert_importance)

    def get_averaged_stats(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        layers = sorted(self.gating_stats.keys())
        if not layers:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        max_experts = max(len(self.gating_stats[l]) for l in layers)
        total_tokens_per_layer = {l: sum(self.token_counts.get(l, [])) for l in layers}

        gating_df = pd.DataFrame(index=[f"expert_{i}" for i in range(max_experts)])
        count_df = pd.DataFrame(index=[f"expert_{i}" for i in range(max_experts)])
        importance_df = pd.DataFrame(index=[f"expert_{i}" for i in range(max_experts)])
        freq_df = pd.DataFrame(index=[f"expert_{i}" for i in range(max_experts)])

        for l in layers:
            tot = total_tokens_per_layer[l]
            gating_df[f"layer_{l}"] = [
                np.sum(self.gating_stats[l][f"expert_{e}"]) / tot if tot else np.nan
                for e in range(max_experts)
            ]
            count_sum_e = [np.sum(self.gating_count[l][f"expert_{e}"]) for e in range(max_experts)]
            count_df[f"layer_{l}"] = count_sum_e
            importance_df[f"layer_{l}"] = [
                np.sum(self.expert_importance_stats[l][f"expert_{e}"]) / count_sum_e[e]
                if count_sum_e[e] > 0 else np.nan
                for e in range(max_experts)
            ]
            freq_df[f"layer_{l}"] = [
                count_sum_e[e] / tot if tot else np.nan
                for e in range(max_experts)
            ]
        return gating_df, count_df, importance_df, freq_df

    def save_to_csv(self, output_path: str) -> None:
        gating_df, count_df, importance_df, freq_df = self.get_averaged_stats()
        base = output_path.rsplit(".", 1)[0]
        gating_df.to_csv(f"{base}_gating.csv")
        count_df.to_csv(f"{base}_gating_count.csv")
        importance_df.to_csv(f"{base}_importance.csv")
        freq_df.to_csv(f"{base}_expert_frequency.csv")


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
        return
    stat = MoEStat(model, tokenizer, device)
    stat.compute_statistics(texts, max_length=max_length)
    stat.save_to_csv(out_path)


def collect_pre_moe_hidden_states_per_layer(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    device,
    max_length: int = 1024,
) -> Dict[int, torch.Tensor]:
    """
    Run model on texts and collect pre-MoE hidden state for each MoE layer.
    Returns dict: layer_idx -> tensor (N_total, hidden_dim).
    """
    layers = model.model.layers
    layer_to_hiddens: Dict[int, List[torch.Tensor]] = {}
    num_moe = 0
    for idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "gate"):
            layer_to_hiddens[idx] = []
            num_moe += 1
    if num_moe == 0:
        return {}

    def make_hook(layer_idx):
        def hook(module, input):
            h = input[0].detach().float()
            layer_to_hiddens[layer_idx].append(h.view(-1, h.shape[-1]))
        return hook

    handles = []
    for layer_idx in layer_to_hiddens:
        mlp = layers[layer_idx].mlp
        handles.append(mlp.register_forward_pre_hook(make_hook(layer_idx)))

    try:
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if inputs["input_ids"].numel() == 0:
                continue
            with torch.no_grad():
                model(**inputs)
        out = {}
        for layer_idx, tensors in layer_to_hiddens.items():
            if tensors:
                out[layer_idx] = torch.cat(tensors, dim=0)
            else:
                out[layer_idx] = None
        return out
    finally:
        for h in handles:
            h.remove()



def main():
    model_path = os.environ.get("MOE_MODEL_PATH", "/data/xiejingyi/WAGLE/files/results/unlearn_wmdp_cyber/GradDiff+Selected/2026-03-29-23-40-09-966832/checkpoints")
    out_dir = os.environ.get("MOE_STAT_OUTPUT", f"/data/xiejingyi/WAGLE/plot/Qwen-GD-Selected-wogating-{os.path.basename(model_path)}-statistics")
    cache_dir = os.environ.get("HF_CACHE", "./.cache")
    num_samples = int(os.environ.get("MOE_NUM_SAMPLES", "25"))
    max_length = int(os.environ.get("MOE_MAX_LENGTH", "1024"))
    os.makedirs(out_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True
    )
    model.seqlen = getattr(model.config, "max_position_embeddings", 1024)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    device = next(model.parameters()).device

    for label, name, subset in DATASET_CONFIGS:
        out_path = os.path.join(out_dir, f"{label}.csv")
        run_one_corpus(name, subset, model, tokenizer, str(device), num_samples, max_length, out_path, cache_dir)

if __name__ == "__main__":
    main()
