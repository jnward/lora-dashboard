# LoRA Activations Dashboard

A simple tool for visualizing LoRA probe activations from language models. Generates a standalone HTML dashboard - no server or dependencies required!

## Project Structure

```
lora-activations-dashboard/
├── backend/                          # Data generation
│   ├── generate_activations_data.py  # Generate activation data from model
│   └── update_dashboard_data.sh      # Convenience script
└── generate_html_dashboard.py        # Generate HTML dashboard from data
```

## Quick Start

### 1. Generate Activation Data

```bash
cd backend
./update_dashboard_data.sh --num-examples 50
```

This will:
- Load the model and LoRA adapters
- Process examples from the dataset
- Extract top-k activations for each layer/projection
- Save data to `activations_data.json`

### 2. Generate HTML Dashboard

```bash
cd ..
python3 generate_html_dashboard.py
```

This creates `lora_activations_dashboard.html` - a standalone HTML file you can open in any browser.

## Usage

### Data Generation Options

```bash
python backend/generate_activations_data.py [options]
```

Options:
- `--base-model`: Base model ID (default: Qwen/Qwen2.5-32B-Instruct)
- `--lora-path`: Path to LoRA checkpoints
- `--num-examples`: Number of examples to process (default: 100)
- `--top-k`: Number of top activations to keep (default: 16)
- `--context-window`: Context window size (default: 10)
- `--no-copy`: Skip automatic copy to frontend

### Dashboard Generation Options

```bash
python generate_html_dashboard.py [options]
```

Options:
- `--data`: Path to activation data JSON (default: backend/activations_data.json)
- `--output`: Output HTML file path (default: lora_activations_dashboard.html)

## Features

- **No Server Required**: Just open the HTML file in your browser (or use the server for interpretations)
- **Interactive Layer Navigation**: Dropdown to switch between layers
- **Token Visualization**: Color-coded activation intensities
  - Red background: Positive activations
  - Blue background: Negative activations
  - Red border: Target token
- **Hover Details**: See exact activation values on hover
- **Responsive Design**: Works on desktop and mobile
- **Side-by-Side Projections**: View gate_proj, up_proj, and down_proj together
- **Feature Interpretations**: Write and save interpretations for each feature
- **Star Important Features**: Mark features of particular interest

## How It Works

1. **Data Generation**: The Python backend loads your model with LoRA adapters and processes examples to find the top-k highest and lowest activating tokens for each layer and projection type.

2. **HTML Generation**: A separate script takes the JSON data and generates a complete HTML file with all data and styling embedded inline.

3. **Visualization**: The dashboard shows token contexts with color intensity proportional to activation magnitude, making it easy to identify patterns in what activates each LoRA neuron.

## Requirements

Backend (data generation):
- PyTorch
- Transformers
- PEFT
- NumPy
- tqdm

Dashboard generation:
- Python 3.x (standard library only)

## Example

```bash
# Generate data for 50 examples
cd backend
./update_dashboard_data.sh --num-examples 50

# Create dashboard
cd ..
python3 generate_html_dashboard.py

# Open in browser
open lora_activations_dashboard.html  # macOS
# or
xdg-open lora_activations_dashboard.html  # Linux
```

## Using Feature Interpretations

The dashboard now supports saving interpretations for each feature. To use this functionality:

### 1. Run the Dashboard Server

```bash
python3 dashboard_server.py
```

This starts a local server on port 8080 that:
- Serves the dashboard at http://localhost:8080
- Saves interpretations to `interpretations.json`
- Provides auto-save functionality

### 2. Write Interpretations

Each of the 6 panels (3 projection types × positive/negative) has:
- **Text area**: Write your interpretation of what the feature detects
- **Star checkbox**: Mark particularly interesting or important features
- **Auto-save**: Interpretations save automatically as you type

### 3. Data Storage

Interpretations are stored in `interpretations.json`:
```json
{
  "interpretations": {
    "layer_10_gate_proj_positive": {
      "text": "Activates on mathematical operators and symbols",
      "starred": true,
      "lastModified": "2024-01-20T10:30:00Z"
    }
  }
}
```

### Server Options

```bash
python3 dashboard_server.py --port 8000 --interpretations-file my_interpretations.json
```

- `--port`: Change server port (default: 8080)
- `--interpretations-file`: Use a different file for storing interpretations