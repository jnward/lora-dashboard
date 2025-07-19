#!/usr/bin/env python3
"""
Generate activation data for LoRA probe analysis dashboard.
This script processes model activations and outputs JSON data for the frontend.
"""

import torch
import torch.nn.functional as F
import glob
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import heapq
import gc
from datetime import datetime
import argparse
import os
import shutil


@dataclass
class ActivationExample:
    """Single activation example with context"""
    activation: float
    rollout_idx: int
    token_idx: int
    token: str
    context: List[str]
    target_position: int
    context_activations: List[float]


@dataclass
class ProjectionData:
    """Data for a single projection type"""
    top_positive: List[ActivationExample]
    top_negative: List[ActivationExample]
    stats: Dict[str, float]


class TopKTracker:
    """Memory-efficient top-k tracker with histogram collection"""
    def __init__(self, k: int, num_bins: int = 50):
        self.k = k
        self.top_positive = []  # min heap
        self.top_negative = []  # max heap (negated)
        self.counter = 0
        self.num_bins = num_bins
        self.all_activations = []  # Collect all activations for histogram
        
    def add(self, activation: float, rollout_idx: int, token_idx: int):
        self.counter += 1
        self.all_activations.append(activation)
        
        if activation >= 0:
            if len(self.top_positive) < self.k:
                heapq.heappush(self.top_positive, (activation, self.counter, rollout_idx, token_idx))
            elif activation > self.top_positive[0][0]:
                heapq.heapreplace(self.top_positive, (activation, self.counter, rollout_idx, token_idx))
        else:
            if len(self.top_negative) < self.k:
                heapq.heappush(self.top_negative, (-activation, self.counter, rollout_idx, token_idx))
            elif -activation > self.top_negative[0][0]:
                heapq.heapreplace(self.top_negative, (-activation, self.counter, rollout_idx, token_idx))
    
    def get_top_positive(self) -> List[Tuple[float, int, int]]:
        return [(act, rid, tid) for act, _, rid, tid in sorted(self.top_positive, key=lambda x: x[0], reverse=True)]
    
    def get_top_negative(self) -> List[Tuple[float, int, int]]:
        return [(-act, rid, tid) for act, _, rid, tid in sorted(self.top_negative, key=lambda x: x[0])]
    
    def compute_histogram(self) -> Dict[str, any]:
        """Compute histogram data for all activations"""
        if not self.all_activations:
            return None
            
        activations_array = np.array(self.all_activations)
        
        # Separate positive and negative for color coding
        positive_acts = activations_array[activations_array >= 0]
        negative_acts = activations_array[activations_array < 0]
        
        # Compute overall histogram range
        min_val = float(np.min(activations_array))
        max_val = float(np.max(activations_array))
        
        # Create bins
        bins = np.linspace(min_val, max_val, self.num_bins + 1)
        
        # Compute histograms
        hist_counts, _ = np.histogram(activations_array, bins=bins)
        pos_counts, _ = np.histogram(positive_acts, bins=bins) if len(positive_acts) > 0 else (np.zeros(self.num_bins), bins)
        neg_counts, _ = np.histogram(negative_acts, bins=bins) if len(negative_acts) > 0 else (np.zeros(self.num_bins), bins)
        
        return {
            'bins': bins.tolist(),
            'total_counts': hist_counts.tolist(),
            'positive_counts': pos_counts.tolist(),
            'negative_counts': neg_counts.tolist(),
            'min': min_val,
            'max': max_val,
            'mean': float(np.mean(activations_array)),
            'std': float(np.std(activations_array)),
            'total_samples': len(self.all_activations)
        }


def extract_probe_directions(model, n_layers: int) -> Tuple[Dict[str, Dict[int, torch.Tensor]], List[int]]:
    """Extract LoRA A matrices (probe directions) from the model
    
    Returns:
        probe_directions: Dict mapping projection types to layer indices to probe directions
        lora_layers: List of layer indices that have LoRA adapters
    """
    probe_directions = {
        'gate_proj': {},
        'up_proj': {},
        'down_proj': {}
    }
    lora_layers = set()
    
    for layer_idx in range(n_layers):
        for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
            module = model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
            
            if hasattr(module, 'lora_A'):
                lora_A_weight = module.lora_A['default'].weight.data
                probe_direction = lora_A_weight.squeeze()
                probe_directions[proj_type][layer_idx] = probe_direction
                lora_layers.add(layer_idx)
    
    return probe_directions, sorted(list(lora_layers))


def compute_lora_cosine_similarities(model, layer_idx: int) -> Dict[str, float]:
    """Compute cosine similarities between LoRA directions in MLP space
    
    For gate_proj and up_proj: uses lora_B (writes to MLP hidden)
    For down_proj: uses lora_A (reads from MLP hidden)
    
    Returns a dict with cosine similarities between all pairs
    """
    directions = {}
    
    # Extract gate_proj and up_proj B matrices (they write to MLP hidden space)
    for proj_type in ['gate_proj', 'up_proj']:
        module = model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
        if hasattr(module, 'lora_B'):
            lora_B_weight = module.lora_B['default'].weight.data
            # lora_B shape is (out_features, r) where r is rank
            # We want the direction it writes to in MLP space
            directions[proj_type] = lora_B_weight.squeeze()
    
    # Extract down_proj A matrix (it reads from MLP hidden space)
    module = model.model.model.layers[layer_idx].mlp.down_proj
    if hasattr(module, 'lora_A'):
        lora_A_weight = module.lora_A['default'].weight.data
        # lora_A shape is (r, in_features) where r is rank
        # We want the direction it reads from in MLP space
        directions['down_proj'] = lora_A_weight.squeeze()
    
    # Compute cosine similarities
    cosine_sims = {}
    proj_types = list(directions.keys())
    
    for i, proj1 in enumerate(proj_types):
        for j, proj2 in enumerate(proj_types):
            if i <= j:  # Only compute upper triangle and diagonal
                dir1 = directions[proj1].float()
                dir2 = directions[proj2].float()
                
                # Normalize and compute cosine similarity
                dir1_norm = dir1 / (dir1.norm() + 1e-8)
                dir2_norm = dir2 / (dir2.norm() + 1e-8)
                
                cosine_sim = torch.dot(dir1_norm, dir2_norm).item()
                cosine_sims[f'{proj1}_{proj2}'] = cosine_sim
                
                # Add symmetric entry if different
                if i != j:
                    cosine_sims[f'{proj2}_{proj1}'] = cosine_sim
    
    return cosine_sims


def process_rollout(model, tokenizer, rollout_data, rollout_idx: int, probe_directions: Dict, 
                   top_k_trackers: Dict, activation_stats: Dict, context_window: int, lora_layers: List[int]):
    """Process a single rollout and update trackers"""
    
    # Extract question and thinking trajectory
    question = rollout_data['question']
    thinking_trajectory = rollout_data.get('deepseek_thinking_trajectory', '')
    attempt = rollout_data.get('deepseek_attempt', '')
    
    if not thinking_trajectory or not attempt:
        return None
    
    # Format the input
    system_prompt = "You are a helpful mathematics assistant."
    full_text = (
        f"<|im_start|>system\n{system_prompt}\n"
        f"<|im_start|>user\n{question}\n"
        f"<|im_start|>assistant\n"
        f"<|im_start|>think\n{thinking_trajectory}\n"
        f"<|im_start|>answer\n{attempt}<|im_end|>"
    )
    
    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids[0]
    
    # Decode tokens
    tokens = []
    for i in range(len(input_ids)):
        decoded = tokenizer.decode(input_ids[i:i+1])
        tokens.append(decoded)
    
    # Storage for activations
    n_layers = model.config.num_hidden_layers
    projected_activations = {
        'gate_proj': {},
        'up_proj': {},
        'down_proj': {}
    }
    
    # Hook functions
    def make_pre_mlp_hook(layer_idx):
        def hook(module, input, output):
            pre_mlp = output.detach()[0]
            for proj_type in ['gate_proj', 'up_proj']:
                probe_dir = probe_directions[proj_type][layer_idx]
                activations = torch.matmul(pre_mlp.float(), probe_dir)
                projected_activations[proj_type][layer_idx] = activations.cpu().numpy()
        return hook
    
    def make_down_proj_hook(layer_idx):
        def hook(module, input, output):
            post_swiglu = input[0].detach()[0]
            probe_dir = probe_directions['down_proj'][layer_idx]
            activations = torch.matmul(post_swiglu.float(), probe_dir)
            projected_activations['down_proj'][layer_idx] = activations.cpu().numpy()
        return hook
    
    # Register hooks only for layers with LoRA adapters
    hooks = []
    for layer_idx in lora_layers:
        layernorm = model.model.model.layers[layer_idx].post_attention_layernorm
        hook = layernorm.register_forward_hook(make_pre_mlp_hook(layer_idx))
        hooks.append(hook)
        
        down_proj = model.model.model.layers[layer_idx].mlp.down_proj
        hook = down_proj.register_forward_hook(make_down_proj_hook(layer_idx))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs.input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Process activations and update trackers
    for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        for layer_idx in lora_layers:
            activations = projected_activations[proj_type][layer_idx]
            
            # Update statistics
            activation_stats[proj_type][layer_idx]['min'] = min(
                activation_stats[proj_type][layer_idx]['min'], 
                float(np.min(activations))
            )
            activation_stats[proj_type][layer_idx]['max'] = max(
                activation_stats[proj_type][layer_idx]['max'], 
                float(np.max(activations))
            )
            
            # Update top-k tracker
            for token_idx in range(len(tokens)):
                activation_value = float(activations[token_idx])
                top_k_trackers[proj_type][layer_idx].add(activation_value, rollout_idx, token_idx)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    return {
        'tokens': tokens,
        'activations': projected_activations
    }


def extract_context_data(tokens: List[str], activations: np.ndarray, token_idx: int, 
                        context_window: int) -> Tuple[List[str], int, List[float]]:
    """Extract context tokens and activations around a target token"""
    context_start = max(0, token_idx - context_window)
    context_end = min(len(tokens), token_idx + context_window + 1)
    
    context_tokens = tokens[context_start:context_end]
    target_position = token_idx - context_start
    context_activations = activations[context_start:context_end].tolist()
    
    return context_tokens, target_position, context_activations


def main(args):
    # Configuration
    print(f"Loading model: {args.base_model}")
    print(f"LoRA path: {args.lora_path}")
    print(f"Processing {args.num_examples} examples")
    print(f"Top-k: {args.top_k}")
    
    # Find LoRA checkpoint
    lora_dirs = glob.glob(f"{args.lora_path}/s1-lora-32B-r{args.rank}-2*")
    if not lora_dirs:
        raise ValueError(f"No LoRA checkpoint found at {args.lora_path}")
    lora_dir = sorted(lora_dirs)[-1]
    print(f"Using LoRA from: {lora_dir}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_dir, torch_dtype=torch.bfloat16)
    
    n_layers = model.config.num_hidden_layers
    
    # Extract probe directions
    print("Extracting probe directions...")
    probe_directions, lora_layers = extract_probe_directions(model, n_layers)
    print(f"Found LoRA adapters in {len(lora_layers)} layers: {lora_layers[:5]}..." if len(lora_layers) > 5 else f"Found LoRA adapters in layers: {lora_layers}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("simplescaling/s1K-1.1", split="train")
    num_examples = min(args.num_examples, len(dataset))
    
    # Initialize trackers only for LoRA layers
    top_k_trackers = {
        proj_type: {layer: TopKTracker(args.top_k, args.histogram_bins) for layer in lora_layers}
        for proj_type in ['gate_proj', 'up_proj', 'down_proj']
    }
    
    activation_stats = {
        proj_type: {layer: {'min': float('inf'), 'max': float('-inf')} for layer in lora_layers}
        for proj_type in ['gate_proj', 'up_proj', 'down_proj']
    }
    
    # Store rollout data for context extraction
    rollout_storage = {}
    
    # Process rollouts
    print(f"Processing {num_examples} rollouts...")
    for rollout_idx in tqdm(range(num_examples), desc="Processing rollouts"):
        rollout = dataset[rollout_idx]
        result = process_rollout(
            model, tokenizer, rollout, rollout_idx, probe_directions,
            top_k_trackers, activation_stats, args.context_window, lora_layers
        )
        
        if result:
            rollout_storage[rollout_idx] = result
        
        # Periodic garbage collection
        if rollout_idx % 10 == 0:
            gc.collect()
    
    print("Extracting final results...")
    
    # Build output data structure
    output_data = {
        "metadata": {
            "modelName": args.base_model,
            "loraPath": lora_dir,
            "numLayers": n_layers,
            "loraLayers": lora_layers,
            "numLoraLayers": len(lora_layers),
            "numExamples": num_examples,
            "topK": args.top_k,
            "contextWindow": args.context_window,
            "generatedAt": datetime.now().isoformat()
        },
        "layers": []
    }
    
    # Process each LoRA layer
    for layer_idx in tqdm(lora_layers, desc="Building output"):
        layer_data = {"layerIdx": layer_idx}
        
        # Compute cosine similarities for this layer
        cosine_sims = compute_lora_cosine_similarities(model, layer_idx)
        layer_data["cosineSimilarities"] = cosine_sims
        
        for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
            # Get top examples
            top_positive = top_k_trackers[proj_type][layer_idx].get_top_positive()
            top_negative = top_k_trackers[proj_type][layer_idx].get_top_negative()
            
            # Build activation examples
            positive_examples = []
            negative_examples = []
            
            for examples_list, output_list in [(top_positive, positive_examples), 
                                              (top_negative, negative_examples)]:
                for activation, rollout_idx, token_idx in examples_list:
                    if rollout_idx not in rollout_storage:
                        continue
                    
                    tokens = rollout_storage[rollout_idx]['tokens']
                    activations = rollout_storage[rollout_idx]['activations'][proj_type][layer_idx]
                    
                    context_tokens, target_position, context_activations = extract_context_data(
                        tokens, activations, token_idx, args.context_window
                    )
                    
                    example = ActivationExample(
                        activation=activation,
                        rollout_idx=rollout_idx,
                        token_idx=token_idx,
                        token=tokens[token_idx],
                        context=context_tokens,
                        target_position=target_position,
                        context_activations=context_activations
                    )
                    output_list.append(asdict(example))
            
            # Compute histogram data
            histogram_data = top_k_trackers[proj_type][layer_idx].compute_histogram()
            
            layer_data[proj_type] = {
                "topPositive": positive_examples,
                "topNegative": negative_examples,
                "stats": {
                    "min": activation_stats[proj_type][layer_idx]['min'],
                    "max": activation_stats[proj_type][layer_idx]['max']
                },
                "histogram": histogram_data
            }
        
        output_data["layers"].append(layer_data)
    
    # Save output
    output_path = args.output or os.path.join(os.path.dirname(__file__), "activations_data.json")
    print(f"Saving data to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Done! Data saved to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # Copy to frontend if not disabled
    if args.copy_to_frontend:
        frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "activations_data.json")
        if os.path.exists(os.path.dirname(frontend_path)):
            print(f"\nCopying to frontend at {frontend_path}...")
            shutil.copy2(output_path, frontend_path)
            print("Data copied to frontend successfully!")
        else:
            print("\nWarning: Frontend directory not found. Skipping copy to frontend.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LoRA activation data for dashboard")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-32B-Instruct", help="Base model ID")
    parser.add_argument("--lora-path", default="/workspace/models/ckpts_1.1", help="Path to LoRA checkpoints")
    parser.add_argument("--rank", type=int, default=1, help="LoRA rank")
    parser.add_argument("--num-examples", type=int, default=100, help="Number of examples to process")
    parser.add_argument("--top-k", type=int, default=16, help="Number of top activations to keep")
    parser.add_argument("--context-window", type=int, default=10, help="Context window size")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--no-copy", dest="copy_to_frontend", action="store_false", 
                       default=True, help="Skip copying to frontend public folder")
    parser.add_argument("--histogram-bins", type=int, default=50, 
                       help="Number of bins for activation histograms")
    
    args = parser.parse_args()
    main(args)