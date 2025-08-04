#!/usr/bin/env python3
"""
Compute logit lens projections for LoRA features.
For down_proj: Projects LoRA B matrices onto unembedding to see output tokens
For gate/up_proj: Projects LoRA A matrices onto embedding to see input tokens
"""

import torch
import json
import glob
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import numpy as np


def extract_lora_matrices(model, n_layers):
    """Extract LoRA A and B matrices for all layers and projections"""
    lora_A_matrices = {'gate_proj': {}, 'up_proj': {}, 'down_proj': {}}
    lora_B_matrices = {'gate_proj': {}, 'up_proj': {}, 'down_proj': {}}
    lora_layers = set()
    
    for layer_idx in range(n_layers):
        for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
            module = model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
            
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Extract A matrix (read direction)
                lora_A_weight = module.lora_A['default'].weight.data
                lora_A_matrices[proj_type][layer_idx] = lora_A_weight.squeeze()
                
                # Extract B matrix (write direction)
                lora_B_weight = module.lora_B['default'].weight.data
                lora_B_matrices[proj_type][layer_idx] = lora_B_weight.squeeze()
                
                lora_layers.add(layer_idx)
    
    return lora_A_matrices, lora_B_matrices, sorted(list(lora_layers))


def compute_logit_lens_projections(model, tokenizer, lora_A_matrices, lora_B_matrices, lora_layers, top_k=20):
    """
    Compute logit lens projections for all LoRA features.
    
    For down_proj: Project LoRA B onto unembedding matrix
    For gate/up_proj: Project LoRA A onto embedding matrix
    """
    # Extract embedding and unembedding matrices
    embed_matrix = model.model.model.embed_tokens.weight.data  # [vocab_size, model_dim]
    unembed_matrix = model.lm_head.weight.data  # [vocab_size, model_dim]
    
    # Get vocabulary
    vocab_size = tokenizer.vocab_size
    
    results = {}
    
    for layer_idx in tqdm(lora_layers, desc="Computing logit lens"):
        results[layer_idx] = {}
        
        for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
            if proj_type == 'down_proj':
                # For down_proj: use LoRA B (write to residual) with unembedding
                if layer_idx not in lora_B_matrices[proj_type]:
                    continue
                    
                direction = lora_B_matrices[proj_type][layer_idx].float()
                matrix = unembed_matrix.float()
                analysis_type = "output"
            else:
                # For gate/up_proj: use LoRA A (read from residual) with embedding
                if layer_idx not in lora_A_matrices[proj_type]:
                    continue
                    
                direction = lora_A_matrices[proj_type][layer_idx].float()
                matrix = embed_matrix.float()
                analysis_type = "input"
            
            # Normalize direction
            direction_norm = direction / (direction.norm() + 1e-8)
            
            # Project onto (un)embedding matrix
            # For embedding: tokens that when embedded have high dot product with direction
            # For unembedding: tokens whose unembedding has high dot product with direction
            logits = torch.matmul(matrix, direction_norm)  # [vocab_size]
            
            # Get top positive and negative tokens
            values, indices = torch.topk(logits, k=min(top_k, vocab_size))
            top_positive_tokens = []
            for i in range(len(values)):
                token_id = indices[i].item()
                value = values[i].item()
                if value <= 0:
                    break
                token = tokenizer.decode([token_id])
                top_positive_tokens.append({
                    'token': token,
                    'token_id': token_id,
                    'value': float(value)
                })
            
            # Get bottom tokens (most negative)
            values, indices = torch.topk(-logits, k=min(top_k, vocab_size))
            top_negative_tokens = []
            for i in range(len(values)):
                token_id = indices[i].item()
                value = -values[i].item()  # Convert back to negative
                if value >= 0:
                    break
                token = tokenizer.decode([token_id])
                top_negative_tokens.append({
                    'token': token,
                    'token_id': token_id,
                    'value': float(value)
                })
            
            results[layer_idx][proj_type] = {
                'analysis_type': analysis_type,
                'top_positive': top_positive_tokens,
                'top_negative': top_negative_tokens,
                'stats': {
                    'max': float(logits.max()),
                    'min': float(logits.min()),
                    'mean': float(logits.mean()),
                    'std': float(logits.std())
                }
            }
    
    return results


def main(args):
    print(f"Loading model: {args.base_model}")
    print(f"LoRA path: {args.lora_path}")
    
    # Find LoRA checkpoint
    lora_dirs = glob.glob(f"{args.lora_path}/s1-lora-32B-r{args.rank}-2*")
    if not lora_dirs:
        raise ValueError(f"No LoRA checkpoint found at {args.lora_path}")
    lora_dir = sorted(lora_dirs)[-1]
    print(f"Using LoRA from: {lora_dir}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
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
    
    # Extract LoRA matrices
    print("Extracting LoRA matrices...")
    lora_A_matrices, lora_B_matrices, lora_layers = extract_lora_matrices(model, n_layers)
    print(f"Found LoRA adapters in {len(lora_layers)} layers")
    
    # Compute logit lens projections
    print("Computing logit lens projections...")
    results = compute_logit_lens_projections(
        model, tokenizer, lora_A_matrices, lora_B_matrices, lora_layers, args.top_k
    )
    
    # Prepare output data
    output_data = {
        'metadata': {
            'model_name': args.base_model,
            'lora_path': lora_dir,
            'num_layers': n_layers,
            'lora_layers': lora_layers,
            'top_k': args.top_k,
            'generated_at': os.path.getmtime(__file__)
        },
        'layers': results
    }
    
    # Save results
    output_path = args.output or os.path.join(os.path.dirname(__file__), "logit_lens_data.json")
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Done! Logit lens data saved to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute logit lens projections for LoRA features")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-32B-Instruct", help="Base model ID")
    parser.add_argument("--lora-path", default="/workspace/models/ckpts_1.1", help="Path to LoRA checkpoints")
    parser.add_argument("--rank", type=int, default=1, help="LoRA rank")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top tokens to keep")
    parser.add_argument("--output", help="Output JSON file path")
    
    args = parser.parse_args()
    main(args)