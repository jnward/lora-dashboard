#!/usr/bin/env python3
"""
Generate a standalone HTML dashboard for LoRA activations.
No server required - just open the HTML file in a browser.
"""

import json
import os
import argparse
import html as html_lib
from datetime import datetime


def generate_histogram_svg(histogram_data, layer_idx, proj_type):
    """Generate SVG histogram for activation distribution"""
    if not histogram_data:
        return ""
    
    bins = histogram_data['bins']
    pos_counts = histogram_data['positive_counts']
    neg_counts = histogram_data['negative_counts']
    
    # SVG dimensions
    width = 300
    height = 150
    margin = {'top': 15, 'right': 15, 'bottom': 30, 'left': 30}
    plot_width = width - margin['left'] - margin['right']
    plot_height = height - margin['top'] - margin['bottom']
    
    # Calculate scales
    max_count = max(max(pos_counts), max(neg_counts)) if (pos_counts and neg_counts) else 1
    x_scale = plot_width / (len(bins) - 1)
    y_scale = plot_height / max_count if max_count > 0 else 1
    
    # Start SVG
    svg = f'<svg class="histogram-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
    svg += f'<g transform="translate({margin["left"]}, {margin["top"]})">'
    
    # Draw bars
    bar_width = x_scale * 0.8
    for i in range(len(pos_counts)):
        x = i * x_scale
        
        # Positive bars (red)
        if pos_counts[i] > 0:
            bar_height = pos_counts[i] * y_scale
            y = plot_height - bar_height
            svg += f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="rgba(255, 0, 0, 0.6)" />'
        
        # Negative bars (blue)
        if neg_counts[i] > 0:
            bar_height = neg_counts[i] * y_scale
            y = plot_height - bar_height
            svg += f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="rgba(0, 0, 255, 0.6)" />'
    
    # Draw axes
    svg += f'<line x1="0" y1="{plot_height}" x2="{plot_width}" y2="{plot_height}" stroke="#333" stroke-width="1" />'
    svg += f'<line x1="0" y1="0" x2="0" y2="{plot_height}" stroke="#333" stroke-width="1" />'
    
    # X-axis labels (show min and max)
    svg += f'<text x="0" y="{plot_height + 20}" text-anchor="middle" font-size="11">{histogram_data["min"]:.2f}</text>'
    svg += f'<text x="{plot_width}" y="{plot_height + 20}" text-anchor="middle" font-size="11">{histogram_data["max"]:.2f}</text>'
    
    # Zero line if range crosses zero
    if histogram_data["min"] < 0 < histogram_data["max"]:
        zero_x = plot_width * (-histogram_data["min"]) / (histogram_data["max"] - histogram_data["min"])
        svg += f'<line x1="{zero_x}" y1="0" x2="{zero_x}" y2="{plot_height}" stroke="#666" stroke-width="1" stroke-dasharray="2,2" />'
        svg += f'<text x="{zero_x}" y="{plot_height + 20}" text-anchor="middle" font-size="11">0</text>'
    
    svg += '</g></svg>'
    
    return svg


def generate_cosine_matrix_html(cosine_sims):
    """Generate HTML table for cosine similarity matrix"""
    if not cosine_sims:
        return ""
    
    proj_types = ['gate_proj', 'up_proj', 'down_proj']
    labels = ['Gate', 'Up', 'Down']
    
    html = '<div style="text-align: center; margin-bottom: 10px; font-size: 0.85em; color: #666;">'
    html += '<span style="display: inline-block; width: 15px; height: 15px; background: rgb(0, 0, 255); vertical-align: middle;"></span> -1.0 '
    html += '<span style="display: inline-block; width: 100px; height: 15px; background: linear-gradient(to right, rgb(0, 0, 255), rgb(255, 255, 255), rgb(255, 0, 0)); vertical-align: middle; margin: 0 5px;"></span>'
    html += ' 1.0 <span style="display: inline-block; width: 15px; height: 15px; background: rgb(255, 0, 0); vertical-align: middle;"></span>'
    html += '</div>'
    html += '<div style="text-align: center;">'
    html += '<table class="matrix-table">'
    
    # Header row
    html += '<tr><th></th>'
    for label in labels:
        html += f'<th class="col-header">{label}</th>'
    html += '</tr>'
    
    # Data rows
    for i, (proj1, label1) in enumerate(zip(proj_types, labels)):
        html += f'<tr><th class="row-header">{label1}</th>'
        for j, (proj2, label2) in enumerate(zip(proj_types, labels)):
            key = f'{proj1}_{proj2}'
            if key in cosine_sims:
                value = cosine_sims[key]
                # Format value
                formatted_value = f'{value:.3f}'
                
                # Calculate color based on value (-1 to 1 range)
                # Map to 0-1 range: 0 = blue (low), 1 = red (high)
                normalized = (value + 1) / 2  # Convert from [-1,1] to [0,1]
                
                # Interpolate between blue and red
                if normalized <= 0.5:
                    # Blue to white
                    intensity = normalized * 2
                    r = int(255 * intensity)
                    g = int(255 * intensity)
                    b = 255
                else:
                    # White to red
                    intensity = (normalized - 0.5) * 2
                    r = 255
                    g = int(255 * (1 - intensity))
                    b = int(255 * (1 - intensity))
                
                bg_color = f'rgb({r}, {g}, {b})'
                style = f'style="background-color: {bg_color};"'
                
                html += f'<td class="matrix-cell" {style}>{formatted_value}</td>'
            else:
                html += '<td class="matrix-cell">-</td>'
        html += '</tr>'
    
    html += '</table></div>'
    return html


def generate_statistics_section(layer_data, layer_idx):
    """Generate collapsible statistics section with cosine similarities and histograms"""
    html = f'<div class="statistics-section" id="stats-{layer_idx}">'
    html += '<div class="statistics-header" onclick="toggleStatistics(' + str(layer_idx) + ')">'
    html += '<span class="statistics-title">Layer Statistics & Distributions</span>'
    html += '<button class="collapse-button collapsed" id="collapse-btn-' + str(layer_idx) + '">▶</button>'
    html += '</div>'
    html += '<div class="statistics-content collapsed" id="stats-content-' + str(layer_idx) + '">'
    html += '<div class="statistics-grid">'
    
    # Cosine similarity matrix
    if 'cosineSimilarities' in layer_data:
        html += '<div class="cosine-matrix-container">'
        html += '<div class="cosine-matrix-title">LoRA Direction Cosine Similarities</div>'
        html += generate_cosine_matrix_html(layer_data['cosineSimilarities'])
        html += '</div>'
    
    # Histograms
    html += '<div class="histograms-container">'
    for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
        if proj_type in layer_data and 'histogram' in layer_data[proj_type] and layer_data[proj_type]['histogram']:
            proj_name = proj_type.upper().replace('_', ' ')
            histogram_data = layer_data[proj_type]['histogram']
            
            html += '<div class="histogram-card">'
            html += f'<div class="histogram-card-title">{proj_name} Distribution</div>'
            html += generate_histogram_svg(histogram_data, layer_idx, proj_type)
            html += '<div class="histogram-stats">'
            html += f'<div class="histogram-stat"><strong>Mean:</strong> {histogram_data["mean"]:.3f}</div>'
            html += f'<div class="histogram-stat"><strong>Std:</strong> {histogram_data["std"]:.3f}</div>'
            html += f'<div class="histogram-stat"><strong>Samples:</strong> {histogram_data["total_samples"]:,}</div>'
            html += '</div></div>'
    
    html += '</div></div></div></div>'
    return html


def generate_token_html(tokens, activations, target_idx):
    """Generate HTML for token context visualization"""
    context_start = 0
    context_end = len(tokens)
    
    html_parts = []
    for i in range(context_start, context_end):
        token = tokens[i]
        activation = activations[i]
        
        # Calculate color intensity
        intensity = min(abs(activation) * 0.05, 0.5)  # Scale down for visibility
        if activation > 0:
            bg_color = f"rgba(255, 0, 0, {intensity})"
        else:
            bg_color = f"rgba(0, 0, 255, {intensity})"
        
        # Escape token and replace newlines, preserve all spaces
        token_display = html_lib.escape(token).replace('\n', '\\n').replace(' ', '&nbsp;')
        
        # Style for target token
        if i == target_idx:
            style = f'style="background-color: {bg_color}; border: 2px solid red; font-weight: bold; padding: 2px 1px; border-radius: 2px; position: relative; display: inline-block;"'
        else:
            style = f'style="background-color: {bg_color}; padding: 2px 1px; border-radius: 2px; position: relative; display: inline-block;"'
        
        html_parts.append(f'<span class="token-with-tooltip" {style}>{token_display}<span class="token-tooltip">{activation:.3f}</span></span>')
    
    return ''.join(html_parts)


def generate_dashboard_html(data_path, output_path):
    """Generate a complete standalone HTML dashboard"""
    
    # Load the activation data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    layers = data['layers']
    
    # Start building HTML
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <title>LoRA Probe Activations Dashboard</title>
    <style>
        /* Reset and base styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.4;
            height: 100vh;
            overflow: hidden;
            margin: 0;
        }
        
        .container {
            width: 100%;
            height: 100vh;
            margin: 0;
            padding: 4px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        /* Compact control bar */
        .control-bar {
            background: white;
            padding: 6px 10px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 15px;
            flex-wrap: nowrap;
        }
        
        .control-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .dashboard-title {
            font-weight: bold;
            color: #1a1a1a;
            font-size: 0.9em;
        }
        
        .layer-controls select {
            padding: 3px 6px;
            border: 1px solid #ddd;
            border-radius: 3px;
            font-size: 13px;
            cursor: pointer;
        }
        
        .nav-button {
            background: #3498db;
            color: white;
            border: none;
            padding: 4px 12px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: background-color 0.2s;
        }
        
        .nav-button.compact {
            padding: 3px 8px;
            font-size: 12px;
        }
        
        .info-item {
            font-size: 0.75em;
            color: #666;
        }
        
        .info-separator {
            color: #ddd;
            font-size: 0.75em;
        }
        
        .legend {
            font-size: 0.7em;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 3px;
            color: #666;
        }
        
        .color-box.small {
            width: 10px;
            height: 10px;
            display: inline-block;
            border: 1px solid #ddd;
        }
        
        .nav-button:hover:not(:disabled) {
            background: #2980b9;
        }
        
        .nav-button:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
        }
        
        /* Layer section */
        .layer-section {
            display: none;
            animation: fadeIn 0.3s ease-in;
            height: 100%;
            flex: 1;
            min-height: 0;
            overflow: hidden;
        }
        
        .layer-section.active {
            display: flex;
            flex-direction: column;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .layer-header {
            display: none;
        }
        
        /* Projections grid */
        .projections-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            flex: 1;
            overflow: hidden;
            min-height: 0;
        }
        
        .projection-column {
            display: flex;
            flex-direction: column;
            gap: 8px;
            min-height: 0;
        }
        
        .projection-card {
            background: white;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }
        
        .projection-header {
            color: white;
            padding: 6px 10px;
            font-weight: bold;
            text-align: center;
            font-size: 0.85em;
        }
        
        .projection-header.positive {
            background: #e74c3c;
        }
        
        .projection-header.negative {
            background: #3498db;
        }
        
        .projection-content {
            padding: 8px;
            overflow-y: auto;
            flex: 1;
            min-height: 0;
        }
        
        /* Activation sections */
        .activation-section {
            margin-bottom: 15px;
        }
        
        .activation-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 6px;
            font-size: 0.85em;
        }
        
        /* Token visualization */
        .token-example {
            background: #f8f9fa;
            padding: 6px 8px;
            border-radius: 3px;
            margin-bottom: 6px;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.75em;
            line-height: 1.4;
            overflow-x: auto;
        }
        
        .example-info {
            font-size: 0.7em;
            color: #666;
            margin-bottom: 3px;
        }
        
        /* Responsive */
        @media (max-width: 1200px) {
            .projections-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        @media (max-width: 768px) {
            .projections-grid {
                grid-template-columns: 1fr;
            }
            
            h1 {
                font-size: 2em;
            }
        }
        
        /* Loading */
        .loading {
            text-align: center;
            padding: 50px;
            color: #666;
        }
        
        
        /* Interpretation section styles */
        .interpretation-section {
            background: #f0f4f8;
            border-top: 1px solid #ddd;
            padding: 8px;
            margin-top: auto;
        }
        
        .interpretation-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        
        .interpretation-title {
            font-weight: bold;
            color: #2c3e50;
            font-size: 0.75em;
        }
        
        .star-container {
            display: flex;
            align-items: center;
            gap: 3px;
        }
        
        .star-checkbox {
            width: 16px;
            height: 16px;
            cursor: pointer;
        }
        
        .star-label {
            color: #666;
            font-size: 0.7em;
            cursor: pointer;
            user-select: none;
        }
        
        .interpretation-textarea {
            width: 100%;
            min-height: 50px;
            padding: 4px 6px;
            border: 1px solid #ddd;
            border-radius: 3px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.75em;
            resize: vertical;
            background: white;
        }
        
        .interpretation-textarea:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        }
        
        .save-status {
            margin-top: 4px;
            font-size: 0.65em;
            color: #666;
            text-align: right;
            min-height: 16px;
        }
        
        .save-status.saving {
            color: #3498db;
        }
        
        .save-status.saved {
            color: #27ae60;
        }
        
        .save-status.error {
            color: #e74c3c;
        }
        
        /* Histogram styles */
        .histogram-svg {
            width: 100%;
            height: 150px;
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
        }
        
        .histogram-stats {
            display: flex;
            justify-content: space-around;
            margin-top: 8px;
            font-size: 0.8em;
            color: #666;
        }
        
        .histogram-stat {
            text-align: center;
        }
        
        .histogram-stat strong {
            color: #333;
        }
        
        /* Tooltip styles */
        .token-with-tooltip {
            position: relative;
            cursor: help;
        }
        
        .token-tooltip {
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s;
            z-index: 1000;
            margin-bottom: 4px;
        }
        
        .token-tooltip::after {
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 4px solid transparent;
            border-top-color: #333;
        }
        
        .token-with-tooltip:hover .token-tooltip {
            opacity: 1;
        }
        
        /* Statistics section styles */
        .statistics-section {
            background: white;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 8px;
            overflow: hidden;
        }
        
        .statistics-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 10px;
            background: #f8f9fa;
            cursor: pointer;
            user-select: none;
        }
        
        .statistics-header:hover {
            background: #e9ecef;
        }
        
        .statistics-title {
            font-weight: bold;
            color: #2c3e50;
            font-size: 0.85em;
        }
        
        .collapse-button {
            background: none;
            border: none;
            font-size: 0.9em;
            color: #6c757d;
            cursor: pointer;
            transition: transform 0.3s;
        }
        
        .collapse-button.collapsed {
            transform: rotate(-90deg);
        }
        
        .statistics-content {
            padding: 20px;
            max-height: 1000px;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }
        
        .statistics-content.collapsed {
            max-height: 0;
            padding: 0 20px;
        }
        
        .statistics-grid {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 20px;
            align-items: start;
        }
        
        .histograms-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }
        
        .histogram-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }
        
        .histogram-card-title {
            font-weight: bold;
            color: #495057;
            margin-bottom: 10px;
            text-align: center;
            font-size: 0.95em;
        }
        
        /* Cosine similarity matrix styles */
        .cosine-matrix-container {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }
        
        .cosine-matrix-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            text-align: center;
            font-size: 1.1em;
        }
        
        .cosine-matrix {
            display: inline-block;
            margin: 0 auto;
            position: relative;
        }
        
        .matrix-table {
            border-collapse: collapse;
            font-size: 0.9em;
            margin: 0 auto;
        }
        
        .matrix-table th {
            padding: 8px 12px;
            text-align: center;
            font-weight: bold;
            color: #2c3e50;
            background: #f8f9fa;
        }
        
        .matrix-table td {
            padding: 8px 12px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }
        
        .matrix-table td.matrix-cell {
            background: white;
            font-weight: bold;
            min-width: 60px;
        }
        
        .matrix-table th.row-header {
            text-align: right;
            background: #f8f9fa;
            border-right: 2px solid #ddd;
        }
        
        .matrix-table th.col-header {
            border-bottom: 2px solid #ddd;
        }
        
    </style>
</head>
<body>
    <div class="container">
        <div class="control-bar">
            <div class="control-group">
                <span class="dashboard-title">LoRA Activations</span>
            </div>
            
            <div class="control-group layer-controls">
                <button class="nav-button compact" id="prev-layer" onclick="navigateLayer(-1)">◀</button>
                <select id="layer-select" onchange="showLayer(this.value)">
                    {layer_options}
                </select>
                <button class="nav-button compact" id="next-layer" onclick="navigateLayer(1)">▶</button>
            </div>
            
            <div class="control-group">
                <span class="info-item">{model_name}</span>
                <span class="info-separator">|</span>
                <span class="info-item">{num_examples} examples</span>
                <span class="info-separator">|</span>
                <span class="info-item">Top-{top_k}</span>
            </div>
            
            <div class="control-group legend">
                <span class="legend-item"><span class="color-box small" style="background: rgba(255, 0, 0, 0.3);"></span>Positive</span>
                <span class="legend-item"><span class="color-box small" style="background: rgba(0, 0, 255, 0.3);"></span>Negative</span>
                <span class="legend-item"><span class="color-box small" style="border: 2px solid red;"></span>Target</span>
            </div>
        </div>
        
        <div id="layers-container" style="flex: 1; min-height: 0; overflow: hidden;">
            {layer_sections}
        </div>
    </div>
    
    <script>
        let layerIndices = [];
        let interpretations = {};
        let saveTimeouts = {};
        
        // API configuration - adjust if running server on different port
        const API_BASE = window.location.port === '8080' ? 'http://localhost:8085' : '';
        
        function toggleStatistics(layerIdx) {
            const content = document.getElementById(`stats-content-${layerIdx}`);
            const button = document.getElementById(`collapse-btn-${layerIdx}`);
            
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                button.classList.remove('collapsed');
                button.textContent = '▼';
            } else {
                content.classList.add('collapsed');
                button.classList.add('collapsed');
                button.textContent = '▶';
            }
        }
        
        async function loadInterpretations() {
            try {
                const response = await fetch(`${API_BASE}/api/interpretations`);
                if (response.ok) {
                    const data = await response.json();
                    interpretations = data.interpretations || {};
                    
                    // Update UI with loaded interpretations
                    Object.keys(interpretations).forEach(featureKey => {
                        const interpretation = interpretations[featureKey];
                        
                        // Update textarea
                        const textarea = document.getElementById(`interpretation-${featureKey}`);
                        if (textarea) {
                            textarea.value = interpretation.text || '';
                        }
                        
                        // Update star checkbox
                        const starCheckbox = document.getElementById(`star-${featureKey}`);
                        if (starCheckbox) {
                            starCheckbox.checked = interpretation.starred || false;
                        }
                    });
                }
            } catch (error) {
                console.error('Failed to load interpretations:', error);
            }
        }
        
        async function saveInterpretation(featureKey, text, starred) {
            const statusElement = document.getElementById(`status-${featureKey}`);
            
            try {
                // Show saving status
                if (statusElement) {
                    statusElement.textContent = 'Saving...';
                    statusElement.className = 'save-status saving';
                }
                
                const response = await fetch(`${API_BASE}/api/interpretations`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        featureKey: featureKey,
                        text: text,
                        starred: starred
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    interpretations[featureKey] = data.interpretation;
                    
                    // Show saved status
                    if (statusElement) {
                        statusElement.textContent = 'Saved';
                        statusElement.className = 'save-status saved';
                        
                        // Clear status after 2 seconds
                        setTimeout(() => {
                            statusElement.textContent = '';
                            statusElement.className = 'save-status';
                        }, 2000);
                    }
                } else {
                    throw new Error('Save failed');
                }
            } catch (error) {
                console.error('Failed to save interpretation:', error);
                
                // Show error status
                if (statusElement) {
                    statusElement.textContent = 'Error saving';
                    statusElement.className = 'save-status error';
                }
            }
        }
        
        function setupInterpretationHandlers() {
            // Handle textarea changes with debouncing
            document.querySelectorAll('.interpretation-textarea').forEach(textarea => {
                textarea.addEventListener('input', (event) => {
                    const featureKey = event.target.dataset.featureKey;
                    const text = event.target.value;
                    const starred = document.getElementById(`star-${featureKey}`).checked;
                    
                    // Clear existing timeout
                    if (saveTimeouts[featureKey]) {
                        clearTimeout(saveTimeouts[featureKey]);
                    }
                    
                    // Set new timeout to save after 500ms of no typing
                    saveTimeouts[featureKey] = setTimeout(() => {
                        saveInterpretation(featureKey, text, starred);
                    }, 500);
                });
            });
            
            // Handle star checkbox changes
            document.querySelectorAll('.star-checkbox').forEach(checkbox => {
                checkbox.addEventListener('change', (event) => {
                    const featureKey = event.target.dataset.featureKey;
                    const starred = event.target.checked;
                    const text = document.getElementById(`interpretation-${featureKey}`).value;
                    
                    saveInterpretation(featureKey, text, starred);
                });
            });
        }
        
        function showLayer(layerIdx) {
            // Hide all layers
            document.querySelectorAll('.layer-section').forEach(section => {
                section.classList.remove('active');
            });
            
            // Show selected layer
            const selectedLayer = document.getElementById(`layer-${layerIdx}`);
            if (selectedLayer) {
                selectedLayer.classList.add('active');
                
                // Set up handlers for newly visible elements
                setTimeout(() => {
                    setupInterpretationHandlers();
                }, 100);
            }
            
            // Update button states
            updateNavigationButtons();
        }
        
        function navigateLayer(direction) {
            const select = document.getElementById('layer-select');
            const currentIndex = select.selectedIndex;
            const newIndex = currentIndex + direction;
            
            if (newIndex >= 0 && newIndex < select.options.length) {
                select.selectedIndex = newIndex;
                showLayer(select.value);
            }
        }
        
        function updateNavigationButtons() {
            const select = document.getElementById('layer-select');
            const currentIndex = select.selectedIndex;
            
            // Update previous button
            const prevButton = document.getElementById('prev-layer');
            prevButton.disabled = currentIndex === 0;
            
            // Update next button
            const nextButton = document.getElementById('next-layer');
            nextButton.disabled = currentIndex === select.options.length - 1;
        }
        
        // Show first layer on load
        window.addEventListener('DOMContentLoaded', async () => {
            const select = document.getElementById('layer-select');
            layerIndices = Array.from(select.options).map(opt => opt.value);
            
            // Load saved interpretations
            await loadInterpretations();
            
            const firstOption = document.querySelector('#layer-select option');
            if (firstOption) {
                showLayer(firstOption.value);
            }
            
            // Set up interpretation handlers
            setupInterpretationHandlers();
        });
    </script>
</body>
</html>"""
    
    # Format metadata
    model_name = metadata['modelName'].split('/')[-1]
    num_lora_layers = metadata.get('numLoraLayers', len(layers))
    layer_indices = metadata.get('loraLayers', [layer['layerIdx'] for layer in layers])
    layer_range = f"{min(layer_indices)}-{max(layer_indices)}"
    num_examples = metadata['numExamples']
    top_k = metadata['topK']
    generated_time = datetime.fromisoformat(metadata['generatedAt']).strftime('%Y-%m-%d %H:%M')
    
    # Generate layer options
    layer_options = '\n'.join([
        f'<option value="{layer["layerIdx"]}">Layer {layer["layerIdx"]}</option>'
        for layer in layers
    ])
    
    # Generate layer sections
    layer_sections = []
    
    for layer in layers:
        layer_idx = layer['layerIdx']
        layer_html = f'<div id="layer-{layer_idx}" class="layer-section">'
        layer_html += f'<h2 class="layer-header">Layer {layer_idx}</h2>'
        
        # Add statistics section
        layer_html += generate_statistics_section(layer, layer_idx)
        
        layer_html += '<div class="projections-grid">'
        
        # Process each projection type - create columns with cells for positive and negative
        for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
            proj_data = layer[proj_type]
            proj_name = proj_type.upper().replace('_', ' ')
            
            # Create a column for this projection type
            layer_html += '<div class="projection-column">'
            
            # Positive activations cell
            layer_html += f'<div class="projection-card">'
            layer_html += f'<div class="projection-header positive">{proj_name} - Positive</div>'
            layer_html += '<div class="projection-content">'
            
            for example in proj_data['topPositive']:
                layer_html += '<div class="token-example">'
                layer_html += f'<div class="example-info">Rollout {example["rollout_idx"]}, Activation: {example["activation"]:.3f}</div>'
                layer_html += '<div>'
                layer_html += generate_token_html(
                    example['context'],
                    example['context_activations'],
                    example['target_position']
                )
                layer_html += '</div></div>'
            
            layer_html += '</div>'
            
            # Add interpretation section for positive
            feature_key = f'layer_{layer_idx}_{proj_type}_positive'
            layer_html += f'''
            <div class="interpretation-section">
                <div class="interpretation-header">
                    <div class="interpretation-title">Interpretation</div>
                    <div class="star-container">
                        <input type="checkbox" class="star-checkbox" id="star-{feature_key}" data-feature-key="{feature_key}">
                        <label for="star-{feature_key}" class="star-label">Star this feature</label>
                    </div>
                </div>
                <textarea class="interpretation-textarea" id="interpretation-{feature_key}" 
                          data-feature-key="{feature_key}" 
                          placeholder="Write your interpretation of this feature..."></textarea>
                <div class="save-status" id="status-{feature_key}"></div>
            </div>
            '''
            layer_html += '</div>'
            
            # Negative activations cell
            layer_html += f'<div class="projection-card">'
            layer_html += f'<div class="projection-header negative">{proj_name} - Negative</div>'
            layer_html += '<div class="projection-content">'
            
            for example in proj_data['topNegative']:
                layer_html += '<div class="token-example">'
                layer_html += f'<div class="example-info">Rollout {example["rollout_idx"]}, Activation: {example["activation"]:.3f}</div>'
                layer_html += '<div>'
                layer_html += generate_token_html(
                    example['context'],
                    example['context_activations'],
                    example['target_position']
                )
                layer_html += '</div></div>'
            
            layer_html += '</div>'
            
            # Add interpretation section for negative
            feature_key = f'layer_{layer_idx}_{proj_type}_negative'
            layer_html += f'''
            <div class="interpretation-section">
                <div class="interpretation-header">
                    <div class="interpretation-title">Interpretation</div>
                    <div class="star-container">
                        <input type="checkbox" class="star-checkbox" id="star-{feature_key}" data-feature-key="{feature_key}">
                        <label for="star-{feature_key}" class="star-label">Star this feature</label>
                    </div>
                </div>
                <textarea class="interpretation-textarea" id="interpretation-{feature_key}" 
                          data-feature-key="{feature_key}" 
                          placeholder="Write your interpretation of this feature..."></textarea>
                <div class="save-status" id="status-{feature_key}"></div>
            </div>
            '''
            layer_html += '</div>'
            
            # Close column
            layer_html += '</div>'
        
        layer_html += '</div></div>'
        layer_sections.append(layer_html)
    
    # Replace placeholders in HTML
    html_content = html_content.replace('{model_name}', model_name)
    html_content = html_content.replace('{num_lora_layers}', str(num_lora_layers))
    html_content = html_content.replace('{layer_range}', layer_range)
    html_content = html_content.replace('{num_examples}', str(num_examples))
    html_content = html_content.replace('{top_k}', str(top_k))
    html_content = html_content.replace('{generated_time}', generated_time)
    html_content = html_content.replace('{layer_options}', layer_options)
    html_content = html_content.replace('{layer_sections}', '\n'.join(layer_sections))
    
    # Write to file
    print(f"Writing dashboard to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Dashboard generated successfully!")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Open {output_path} in your browser to view the dashboard.")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML dashboard from activation data")
    parser.add_argument("--data", default="backend/activations_data.json", 
                       help="Path to activation data JSON file")
    parser.add_argument("--output", default="lora_activations_dashboard.html",
                       help="Output HTML file path")
    
    args = parser.parse_args()
    
    # Find the data file
    if not os.path.exists(args.data):
        # Try common locations
        possible_paths = [
            "backend/activations_data.json",
            "frontend/public/activations_data.json",
            "activations_data.json"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                args.data = path
                break
        else:
            print(f"Error: Could not find activation data file at {args.data}")
            print("Please run generate_activations_data.py first or specify --data path")
            return 1
    
    generate_dashboard_html(args.data, args.output)
    return 0


if __name__ == "__main__":
    exit(main())