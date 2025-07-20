# LoRA Activations Dashboard Project

## Overview

This project analyzes and visualizes rank-1 LoRA (Low-Rank Adaptation) activations to interpret learned reasoning directions in language models. It provides insights into how LoRA adapters modify model behavior by examining neuron-level activations across different layers and projection types.

## Core Concept: Rank-1 LoRA Activations

### What are LoRA Activations?
A rank-1 LoRA "activation" is the scalar value computed from the dot product between:
- Input features and the LoRA A matrix (the "read" direction)
- This scalar is then used to scale the LoRA B matrix (the "write" direction)

For a rank-1 LoRA, this creates a single scalar activation value per token that determines how strongly the LoRA adapter fires.

### Model Architecture
- **Base Model**: Qwen-2.5-32B-Instruct
- **LoRA Configuration**: Rank-1 adapters trained on R1 reasoning traces
- **Adapter Locations**: MLP layers only (gate_proj, up_proj, down_proj)

## Technical Implementation

### Data Generation Pipeline (`generate_activations_data.py`)

1. **Model Loading**: Loads base model with LoRA adapters
2. **Probe Direction Extraction**: Extracts LoRA A matrices as probe directions
3. **Activation Computation**:
   - For `gate_proj` and `up_proj`: Computes activations at pre-MLP layer (after post_attention_layernorm)
   - For `down_proj`: Computes activations at post-SwiGLU (input to down_proj)
4. **Top-K Tracking**: Maintains memory-efficient tracking of highest/lowest activations
5. **Statistical Analysis**: Computes histograms and cosine similarities between directions

### Key Features

#### 1. Activation Analysis
- **Top-K Examples**: Tracks the most positive and negative activating tokens
- **Context Windows**: Shows surrounding tokens with their activation values
- **Token-Level Visualization**: Color-coded activation intensities (red=positive, blue=negative)

#### 2. Statistical Insights
- **Activation Distributions**: Histograms showing activation patterns across the dataset
- **Cosine Similarities**: Measures relationships between gate/up/down projection directions in MLP hidden space
- **Layer Statistics**: Min/max values, means, and standard deviations

#### 3. Interpretability Tools
- **Max-Activating Examples**: Identifies tokens/contexts that most strongly activate each direction
- **Pattern Detection**: Helps identify what linguistic features each LoRA neuron responds to
- **Cross-Layer Analysis**: Navigate between layers to see how patterns evolve

## Dashboard Features

### Interactive HTML Visualization
- **No Server Required**: Standalone HTML file with embedded data
- **Layer Navigation**: Easy switching between LoRA-adapted layers
- **Responsive Design**: Works on desktop and mobile devices

### Visual Elements
1. **Token Visualization**:
   - Color intensity shows activation magnitude
   - Red border highlights target token
   - Hover tooltips display exact activation values

2. **Statistical Displays**:
   - Cosine similarity matrices between projection types
   - Activation distribution histograms
   - Collapsible statistics sections

3. **Organization**:
   - Side-by-side comparison of gate_proj, up_proj, down_proj
   - Separate sections for positive and negative activations

## Usage Workflow

1. **Generate Activation Data**:
   ```bash
   python backend/generate_activations_data.py --num-examples 100 --top-k 16
   ```

2. **Create Dashboard**:
   ```bash
   python generate_html_dashboard.py
   ```

3. **Analyze Results**:
   - Open the HTML file in a browser
   - Navigate between layers
   - Examine max-activating examples to understand what each LoRA neuron detects

## Interpretation Guidelines

### Understanding Activations
- **High Positive Activations**: The LoRA strongly amplifies its learned direction
- **High Negative Activations**: The LoRA suppresses or inverts its learned direction
- **Near-Zero Activations**: The LoRA has minimal effect

### Cosine Similarities
- **Between gate_proj and up_proj**: Shows if gating and value computations align
- **With down_proj**: Indicates if the readout direction aligns with input directions

### Common Patterns
- Look for semantic coherence in max-activating examples
- Check if activations correlate with specific reasoning steps
- Identify whether adapters specialize in particular token types or positions

## Technical Notes

### Memory Efficiency
- Uses heap-based top-k tracking to handle large datasets
- Processes examples incrementally with garbage collection
- Stores only essential context for visualization

### Coordinate System
- All activations are computed in the model's native representation
- Positive/negative values indicate direction along the learned LoRA axis
- Magnitude indicates strength of the adapter's influence