# %% [markdown]
# # LoRA Steering Experiments
# 
# This notebook implements steering experiments for rank-1 LoRA adapters.
# We'll artificially increase LoRA activations to see their effect on model generation.

# %%
import torch
import torch.nn.functional as F
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import glob
from tqdm import tqdm
import random
import gc
import h5py
import os
from pathlib import Path

# %% [markdown]
# ## Configuration

# %%
# Model configuration
BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
LORA_PATH = "/workspace/models/ckpts_1.1"
RANK = 1

# Steering configuration - will be multiplied by median non-zero activation
# Log-spaced multipliers from 256 to 4096 (8 values)
# np.logspace(np.log10(256), np.log10(4096), 8) gives:
STEERING_MULTIPLIERS = [128, 196, 256, 384, 576, 864, 1296, 1944, 2916, 4096]
MAX_NEW_TOKENS = 16
TEMPERATURE = 0.0
TOP_P = 0.95
NUM_SENTENCES_PREFIX = 5  # Generate 5 sentences as prefix

# Paths
ACTIVATIONS_DIR = "/workspace/lora-activations-dashboard/backend/activations"
MEDIAN_CACHE_PATH = "/workspace/lora-activations-dashboard/median_activations_cache.json"

# %% [markdown]
# ## Load Model and Dataset

# %%
# Find LoRA checkpoint
lora_dirs = glob.glob(f"{LORA_PATH}/s1-lora-32B-r{RANK}-2*544")
if not lora_dirs:
    raise ValueError(f"No LoRA checkpoint found at {LORA_PATH}")
lora_dir = sorted(lora_dirs)[-1]
print(f"Using LoRA from: {lora_dir}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_dir, torch_dtype=torch.bfloat16)

# Load dataset
print("Loading dataset...")
dataset = load_dataset("simplescaling/s1K-1.1", split="train")

# %% [markdown]
# ## Extract LoRA Directions

# %%
def extract_lora_directions(model):
    """Extract LoRA A and B matrices for all layers and projections"""
    lora_A_directions = {'gate_proj': {}, 'up_proj': {}, 'down_proj': {}}
    lora_B_directions = {'gate_proj': {}, 'up_proj': {}, 'down_proj': {}}
    lora_layers = set()
    
    n_layers = model.config.num_hidden_layers
    
    for layer_idx in range(n_layers):
        for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
            module = model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
            
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Extract A matrix (read direction)
                lora_A_weight = module.lora_A['default'].weight.data
                lora_A_directions[proj_type][layer_idx] = lora_A_weight.squeeze()
                
                # Extract B matrix (write direction)
                lora_B_weight = module.lora_B['default'].weight.data
                lora_B_directions[proj_type][layer_idx] = lora_B_weight.squeeze()
                
                lora_layers.add(layer_idx)
    
    return lora_A_directions, lora_B_directions, sorted(list(lora_layers))

# Extract directions
print("Extracting LoRA directions...")
lora_A_directions, lora_B_directions, lora_layers = extract_lora_directions(model)
print(f"Found LoRA adapters in {len(lora_layers)} layers")

# %% [markdown]
# ## Compute Median Non-Zero Activations

# %%
def compute_median_nonzero_activations(activations_dir, lora_layers, cache_path=None):
    """
    Compute median non-zero activations for each feature from HDF5 files.
    
    Returns a dict: {layer_idx: {proj_type: {polarity: median_value}}}
    """
    # Try to load from cache first
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached median activations from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f, object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()})
    
    # Check if activation files exist
    h5_files = glob.glob(os.path.join(activations_dir, "rollout_*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No activation files found in {activations_dir}. Please run generate_activations_data.py first.")
    
    print(f"Computing median non-zero activations from {len(h5_files)} files...")
    
    # Initialize storage
    activations_by_feature = {}
    proj_types = ['gate_proj', 'up_proj', 'down_proj']
    
    # Process each file
    for h5_path in tqdm(h5_files[:100], desc="Processing activation files"):  # Use first 100 files for efficiency
        with h5py.File(h5_path, 'r') as f:
            # activations shape: [num_tokens, num_layers, 3]
            activations = f['activations'][:]
            
            for layer_idx_pos, layer_idx in enumerate(lora_layers):
                if layer_idx not in activations_by_feature:
                    activations_by_feature[layer_idx] = {proj: {'positive': [], 'negative': []} for proj in proj_types}
                
                for proj_idx, proj_type in enumerate(proj_types):
                    layer_activations = activations[:, layer_idx_pos, proj_idx]
                    
                    # Separate positive and negative non-zero activations
                    positive_acts = layer_activations[layer_activations > 0]
                    negative_acts = layer_activations[layer_activations < 0]
                    
                    if len(positive_acts) > 0:
                        activations_by_feature[layer_idx][proj_type]['positive'].extend(positive_acts.tolist())
                    if len(negative_acts) > 0:
                        activations_by_feature[layer_idx][proj_type]['negative'].extend(negative_acts.tolist())
    
    # Compute medians
    median_activations = {}
    for layer_idx in lora_layers:
        median_activations[layer_idx] = {}
        for proj_type in proj_types:
            median_activations[layer_idx][proj_type] = {}
            
            # Positive median
            pos_acts = activations_by_feature[layer_idx][proj_type]['positive']
            if pos_acts:
                median_activations[layer_idx][proj_type]['positive'] = float(np.median(pos_acts))
            else:
                median_activations[layer_idx][proj_type]['positive'] = 1.0  # Default if no positive activations
            
            # Negative median (store as positive value for magnitude)
            neg_acts = activations_by_feature[layer_idx][proj_type]['negative']
            if neg_acts:
                median_activations[layer_idx][proj_type]['negative'] = float(np.median(np.abs(neg_acts)))
            else:
                median_activations[layer_idx][proj_type]['negative'] = 1.0  # Default if no negative activations
    
    # Save cache
    if cache_path:
        print(f"Saving median activations cache to {cache_path}")
        with open(cache_path, 'w') as f:
            json.dump(median_activations, f, indent=2)
    
    return median_activations

# Compute median activations
print("\nComputing median non-zero activations...")
median_activations = compute_median_nonzero_activations(ACTIVATIONS_DIR, lora_layers, MEDIAN_CACHE_PATH)

# Print some examples
print("\nExample median activations:")
for layer_idx in lora_layers[:3]:
    print(f"\nLayer {layer_idx}:")
    for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        pos_med = median_activations[layer_idx][proj_type]['positive']
        neg_med = median_activations[layer_idx][proj_type]['negative']
        print(f"  {proj_type}: positive={pos_med:.4f}, negative={neg_med:.4f}")

# %% [markdown]
# ## Select a Random Prompt

# %%
# Select a random prompt from the dataset
random_idx = random.randint(0, len(dataset) - 1)
selected_prompt = dataset[random_idx]['question']
print(f"Selected prompt (idx {random_idx}):")
print(selected_prompt)

# %% [markdown]
# ## Compute Standard Deviations of LoRA Activations

# %%
def compute_lora_activation_stats(activations_dir, lora_layers, cache_path=None):
    """
    Compute standard deviations of LoRA activations for each feature.
    This allows steering strength to be measured in units of standard deviations.
    
    Returns:
        dict: {layer_idx: {proj_type: {"positive": stats, "negative": stats}}}
    """
    # Try to load from cache first
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached LoRA activation stats from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f, object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()})
    
    # Check if activation files exist
    h5_files = glob.glob(os.path.join(activations_dir, "rollout_*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No activation files found in {activations_dir}. Please run generate_activations_data.py first.")
    
    print(f"Computing LoRA activation statistics from {len(h5_files)} files...")
    
    # Initialize storage
    activations_by_feature = {}
    proj_types = ['gate_proj', 'up_proj', 'down_proj']
    
    # Process each file
    for h5_path in tqdm(h5_files[:100], desc="Processing activation files"):  # Use first 100 files
        with h5py.File(h5_path, 'r') as f:
            # activations shape: [num_tokens, num_layers, 3]
            activations = f['activations'][:]
            
            for layer_idx_pos, layer_idx in enumerate(lora_layers):
                if layer_idx not in activations_by_feature:
                    activations_by_feature[layer_idx] = {proj: {'positive': [], 'negative': []} for proj in proj_types}
                
                for proj_idx, proj_type in enumerate(proj_types):
                    layer_activations = activations[:, layer_idx_pos, proj_idx]
                    
                    # Collect all non-zero activations
                    positive_acts = layer_activations[layer_activations > 0]
                    negative_acts = layer_activations[layer_activations < 0]
                    
                    if len(positive_acts) > 0:
                        activations_by_feature[layer_idx][proj_type]['positive'].extend(positive_acts.tolist())
                    if len(negative_acts) > 0:
                        activations_by_feature[layer_idx][proj_type]['negative'].extend(np.abs(negative_acts).tolist())
    
    # Compute statistics
    lora_stats = {}
    for layer_idx in lora_layers:
        lora_stats[layer_idx] = {}
        for proj_type in proj_types:
            lora_stats[layer_idx][proj_type] = {}
            
            # Positive stats
            pos_acts = activations_by_feature[layer_idx][proj_type]['positive']
            if pos_acts and len(pos_acts) > 1:
                lora_stats[layer_idx][proj_type]['positive'] = {
                    'std': float(np.std(pos_acts)),
                    'mean': float(np.mean(pos_acts)),
                    'median': float(np.median(pos_acts))
                }
            else:
                lora_stats[layer_idx][proj_type]['positive'] = {'std': 1.0, 'mean': 1.0, 'median': 1.0}
            
            # Negative stats (using absolute values)
            neg_acts = activations_by_feature[layer_idx][proj_type]['negative']
            if neg_acts and len(neg_acts) > 1:
                lora_stats[layer_idx][proj_type]['negative'] = {
                    'std': float(np.std(neg_acts)),
                    'mean': float(np.mean(neg_acts)),
                    'median': float(np.median(neg_acts))
                }
            else:
                lora_stats[layer_idx][proj_type]['negative'] = {'std': 1.0, 'mean': 1.0, 'median': 1.0}
    
    # Save cache
    if cache_path:
        print(f"Saving LoRA activation stats to {cache_path}")
        with open(cache_path, 'w') as f:
            json.dump(lora_stats, f, indent=2)
    
    return lora_stats

# Compute LoRA activation statistics
print("\nComputing LoRA activation statistics...")
lora_stats = compute_lora_activation_stats(ACTIVATIONS_DIR, lora_layers, 
                                          "/workspace/lora-activations-dashboard/lora_activation_stats_cache.json")

# Print some examples
print("\nExample LoRA activation statistics:")
for layer_idx in lora_layers[:3]:
    print(f"\nLayer {layer_idx}:")
    for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        pos_stats = lora_stats[layer_idx][proj_type]['positive']
        neg_stats = lora_stats[layer_idx][proj_type]['negative']
        print(f"  {proj_type}:")
        print(f"    Positive: median={pos_stats['median']:.4f}, std={pos_stats['std']:.4f}")
        print(f"    Negative: median={neg_stats['median']:.4f}, std={neg_stats['std']:.4f}")

# %% [markdown]
# ## Generate Prefix Without Steering

# %%
def count_sentences(text):
    """Simple sentence counter based on common punctuation"""
    import re
    # Count sentences ending with ., !, or ?
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty strings and count
    return len([s for s in sentences if s.strip()])

def generate_prefix(model, tokenizer, prompt, num_sentences=5):
    """Generate a prefix with approximately num_sentences using greedy decoding"""
    system_prompt = "You are a helpful mathematics assistant."
    full_prompt = (
        f"<|im_start|>system\n{system_prompt}\n"
        f"<|im_start|>user\n{prompt}\n"
        f"<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    # Generate token by token until we have enough sentences
    generated_ids = inputs.input_ids[0].tolist()
    
    with torch.no_grad():
        for _ in range(500):  # Max tokens to prevent infinite loop
            input_tensor = torch.tensor([generated_ids]).to(model.device)
            
            outputs = model(input_tensor)
            logits = outputs.logits[0, -1, :]
            
            # Greedy decoding: take the argmax
            next_token = torch.argmax(logits).item()
            
            generated_ids.append(next_token)
            
            # Check if we have enough sentences
            generated_text = tokenizer.decode(generated_ids[len(inputs.input_ids[0]):])
            if count_sentences(generated_text) >= num_sentences:
                # Find the end of the current sentence
                for _ in range(50):  # Look ahead up to 50 tokens
                    if tokenizer.decode([next_token]) in '.!?':
                        break
                    
                    input_tensor = torch.tensor([generated_ids]).to(model.device)
                    outputs = model(input_tensor)
                    logits = outputs.logits[0, -1, :]
                    next_token = torch.argmax(logits).item()
                    generated_ids.append(next_token)
                break
    
    # Return full prompt with generated prefix
    full_text = tokenizer.decode(generated_ids)
    prefix_only = tokenizer.decode(generated_ids[len(inputs.input_ids[0]):])
    
    return full_text, prefix_only

# Generate prefix
print("\nGenerating prefix (5 sentences)...")
prefix_with_prompt, prefix_only = generate_prefix(model, tokenizer, selected_prompt, NUM_SENTENCES_PREFIX)
print(f"\nGenerated prefix:\n{prefix_only}")

# %% [markdown]
# ## Implement Steering Hooks

# %%
def create_steering_hooks(model, layer_idx, proj_type, polarity, magnitude, lora_B_directions, lora_stats):
    """Create hooks for steering based on projection type and polarity
    
    Args:
        model: The model
        layer_idx: Layer index
        proj_type: 'gate_proj', 'up_proj', or 'down_proj'
        polarity: 1 for positive, -1 for negative
        magnitude: Steering magnitude (already scaled by median activation)
        lora_B_directions: Dict of LoRA B matrices
        lora_stats: Dict of LoRA activation statistics for each layer
    """
    hooks = []
    
    # Get the B matrix for this layer/projection
    if layer_idx not in lora_B_directions[proj_type]:
        return hooks
    
    lora_B = lora_B_directions[proj_type][layer_idx]
    
    # Apply polarity and magnitude (no additional scaling)
    steering_vector = lora_B * polarity * magnitude
    
    if proj_type in ['gate_proj', 'up_proj']:
        # For gate_proj and up_proj, we add to the output of the projection
        # This affects the MLP hidden state
        module = model.model.model.layers[layer_idx].mlp.__getattr__(proj_type)
        
        def mlp_steering_hook(module, input, output):
            # output shape: [batch, seq_len, hidden_dim]
            output[0, :, :] = output[0, :, :] + steering_vector.to(output.dtype).to(output.device)
            return output
        
        hook = module.register_forward_hook(mlp_steering_hook)
        hooks.append(hook)
        
    elif proj_type == 'down_proj':
        # For down_proj, we add to the output of down_proj (which goes to residual stream)
        module = model.model.model.layers[layer_idx].mlp.down_proj
        
        def residual_steering_hook(module, input, output):
            # output shape: [batch, seq_len, model_dim]
            output[0, :, :] = output[0, :, :] + steering_vector.to(output.dtype).to(output.device)
            return output
        
        hook = module.register_forward_hook(residual_steering_hook)
        hooks.append(hook)
    
    return hooks

# %% [markdown]
# ## Test Steering on Multiple Features

# %%
# Configuration for which features to test
# Option 1: Test specific layer/projection combinations
TEST_FEATURES = [
    (10, 'up_proj'),
    (20, 'down_proj'),
    (30, 'down_proj'),
    (40, 'down_proj'),
    (45, 'down_proj'),
]

# Option 2: Test multiple layers with same projection
# TEST_LAYERS = [20, 30, 40, 45, 50]
# TEST_PROJECTION = 'gate_proj'
# TEST_FEATURES = [(layer, TEST_PROJECTION) for layer in TEST_LAYERS]

# Option 3: Test all combinations
# TEST_LAYERS = [20, 30, 40]
# TEST_PROJECTIONS = ['gate_proj', 'up_proj', 'down_proj']
# TEST_FEATURES = [(layer, proj) for layer in TEST_LAYERS for proj in TEST_PROJECTIONS]

all_test_results = {}

# Test each feature
for test_layer, test_proj in TEST_FEATURES:
    feature_key = f"layer_{test_layer}_{test_proj}"
    all_test_results[feature_key] = {}
    
    print(f"\n{'='*60}")
    print(f"TESTING FEATURE: Layer {test_layer}, {test_proj}")
    print(f"{'='*60}")
    
    # Test both polarities
    for test_polarity in [1, -1]:  # positive and negative
        polarity_name = "positive" if test_polarity == 1 else "negative"
        print(f"\n{'='*50}")
        print(f"Testing polarity={polarity_name}")
        print(f"{'='*50}")
        
        # Get median activation for this feature
        median_act = median_activations[test_layer][test_proj][polarity_name]
        print(f"Median non-zero activation: {median_act:.4f}")
        print("Generating with different magnitudes...")
        
        test_results = {}
        
        for multiplier in STEERING_MULTIPLIERS:
            # Calculate magnitude based on median activation
            magnitude = multiplier * median_act
            print(f"\nMultiplier {multiplier}x (magnitude={magnitude:.4f}):")
            
            # Register steering hooks
            hooks = create_steering_hooks(model, test_layer, test_proj, test_polarity, magnitude, lora_B_directions, lora_stats)
            
            # Tokenize the prefix
            inputs = tokenizer(prefix_with_prompt, return_tensors="pt").to(model.device)
            
            # Generate with greedy decoding
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Decode and store
            generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            test_results[multiplier] = generated_text
            
            print(generated_text[:100] + "..." if len(generated_text) > 100 else generated_text)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
        
        all_test_results[feature_key][polarity_name] = test_results

# %% [markdown]
# ## Analyze Results

# %%
print("\n=== STEERING RESULTS COMPARISON ===\n")
print(f"Original prompt: {selected_prompt}\n")
print(f"Prefix: {prefix_only}\n")

# Analyze results for each feature
for test_layer, test_proj in TEST_FEATURES:
    feature_key = f"layer_{test_layer}_{test_proj}"
    
    print(f"\n{'='*60}")
    print(f"FEATURE: Layer {test_layer}, {test_proj}")
    print(f"{'='*60}")
    
    for polarity_name in ["positive", "negative"]:
        print(f"\n{'='*50}")
        print(f"POLARITY: {polarity_name.upper()}")
        print(f"{'='*50}")
        
        # Show median activation for this polarity
        median_act = median_activations[test_layer][test_proj][polarity_name]
        print(f"Median non-zero activation: {median_act:.4f}")
        
        for multiplier in STEERING_MULTIPLIERS:
            magnitude = multiplier * median_act
            print(f"\n--- Multiplier {multiplier}x (magnitude={magnitude:.4f}) ---")
            print(all_test_results[feature_key][polarity_name][multiplier])

# %% [markdown]
# ## Save Test Results

# %%
# Save the test results
test_output = {
    "prompt": selected_prompt,
    "prefix": prefix_only,
    "tested_features": TEST_FEATURES,
    "multipliers": STEERING_MULTIPLIERS,
    "features": {}
}

# Add data for each tested feature
for test_layer, test_proj in TEST_FEATURES:
    feature_key = f"layer_{test_layer}_{test_proj}"
    test_output["features"][feature_key] = {
        "layer": test_layer,
        "projection": test_proj,
        "median_activations": {
            "positive": median_activations[test_layer][test_proj]["positive"],
            "negative": median_activations[test_layer][test_proj]["negative"]
        },
        "lora_stats": {
            "positive": lora_stats[test_layer][test_proj]["positive"],
            "negative": lora_stats[test_layer][test_proj]["negative"]
        },
        "results": all_test_results[feature_key]
    }

with open("steering_test_results.json", "w") as f:
    json.dump(test_output, f, indent=2)

print("\nTest results saved to steering_test_results.json")
# %%