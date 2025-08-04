#!/usr/bin/env python3
"""
Generate a feature selection HTML dashboard for interpreting specific LoRA features.
Unlike the interpretation dashboard, this allows manual selection of features to analyze.
"""

import json
import os
import argparse
import html as html_lib
from datetime import datetime


def generate_token_html(tokens, activations, target_idx, context_window=10):
    """Generate HTML for token context visualization"""
    context_start = max(0, target_idx - context_window)
    context_end = min(len(tokens), target_idx + context_window + 1)
    
    html_parts = []
    for i in range(context_start, context_end):
        token = tokens[i]
        activation = activations[i]
        
        # Calculate color intensity
        intensity = min(abs(activation) * 0.1, 0.7)
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
    """Generate the feature selection dashboard"""
    
    # Load the activation data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    layers = data['layers']
    
    # Build list of all features
    all_features = []
    for layer_data in layers:
        layer_idx = layer_data['layerIdx']
        for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
            if proj_type in layer_data:
                # Add positive feature (one per projection type)
                feature_key = f'layer_{layer_idx}_{proj_type}_positive'
                all_features.append({
                    'key': feature_key,
                    'layer': layer_idx,
                    'projection': proj_type,
                    'polarity': 'positive',
                    'examples': layer_data[proj_type]['topPositive']
                })
                
                # Add negative feature (one per projection type)
                feature_key = f'layer_{layer_idx}_{proj_type}_negative'
                all_features.append({
                    'key': feature_key,
                    'layer': layer_idx,
                    'projection': proj_type,
                    'polarity': 'negative',
                    'examples': layer_data[proj_type]['topNegative']
                })
    
    # Count total features
    total_features = len(all_features)
    
    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <title>LoRA Feature Selection Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 0;
            margin: 0;
            overflow: hidden;
        }}
        
        .main-layout {{
            display: flex;
            height: 100vh;
            width: 100vw;
            padding-bottom: 80px; /* Space for control bar */
            box-sizing: border-box;
        }}
        
        .left-panel {{
            flex: 0 0 40%;
            padding: 20px;
            overflow-y: auto;
            padding-bottom: 20px; /* Normal bottom padding */
        }}
        
        .right-panel {{
            flex: 0 0 60%;
            background: white;
            border-left: 1px solid #ddd;
            position: relative;
            display: none; /* Hidden initially, will be changed to flex when shown */
            flex-direction: column;
        }}
        
        .context-wrapper {{
            position: relative;
            flex: 1;
            display: flex;
            overflow: hidden;
        }}
        
        .container {{
            max-width: 100%;
            margin: 0 auto;
        }}
        
        /* Feature selection section */
        .feature-selection-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .selection-title {{
            font-size: 1.4em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c3e50;
        }}
        
        .selection-controls {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .control-label {{
            font-weight: 600;
            color: #555;
            font-size: 0.95em;
        }}
        
        .control-select {{
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 1em;
            background: white;
            cursor: pointer;
            transition: border-color 0.2s;
        }}
        
        .control-select:focus {{
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        }}
        
        .load-feature-button {{
            padding: 12px 24px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
            align-self: flex-start;
        }}
        
        .load-feature-button:hover {{
            background: #2980b9;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        /* Progress section */
        .progress-section {{
            background: white;
            padding: 10px 15px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }}
        
        .progress-title {{
            font-size: 0.9em;
            font-weight: bold;
            margin-bottom: 6px;
            color: #2c3e50;
        }}
        
        .progress-bar-container {{
            background: #e0e0e0;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }}
        
        .progress-bar {{
            height: 100%;
            background: linear-gradient(to right, #3498db, #2ecc71);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        
        .progress-stats {{
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
            font-size: 0.9em;
            color: #666;
        }}
        
        /* Feature display */
        .feature-section {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .feature-header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .feature-title {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .feature-subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        
        /* Examples */
        .examples-container {{
            margin-bottom: 30px;
        }}
        
        .example-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.9em;
            line-height: 1.8;
            overflow-x: auto;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .example-item:hover {{
            background: #e9ecef;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .example-item.selected {{
            background: #d4edda;
            border: 2px solid #28a745;
        }}
        
        .example-info {{
            font-size: 0.85em;
            color: #666;
            margin-bottom: 8px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        
        
        /* Control section */
        .control-section {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 80px; /* Explicit height */
            background: white;
            padding: 15px 20px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 30px;
            z-index: 1000;
            box-sizing: border-box; /* Include padding in height */
        }}
        
        .interpretation-mini {{
            display: flex;
            align-items: center;
            gap: 15px;
            flex: 0 1 600px;
        }}
        
        .interpretation-mini-textarea {{
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.95em;
            resize: none;
            height: 50px;
            background: white;
        }}
        
        .interpretation-mini-textarea:focus {{
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        }}
        
        .star-container-mini {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .button-group {{
            display: flex;
            gap: 15px;
        }}
        
        .control-button {{
            padding: 12px 30px;
            font-size: 1.1em;
            font-weight: bold;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .save-button {{
            background: #2ecc71;
            color: white;
        }}
        
        .save-button:hover {{
            background: #27ae60;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        .save-skip-button {{
            background: #e74c3c;
            color: white;
        }}
        
        .save-skip-button:hover {{
            background: #c0392b;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        /* Completion message */
        .completion-message {{
            text-align: center;
            padding: 50px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .completion-title {{
            font-size: 2em;
            color: #2ecc71;
            margin-bottom: 20px;
        }}
        
        /* Tooltip styles */
        .token-with-tooltip {{
            position: relative;
            cursor: help;
        }}
        
        .token-tooltip {{
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
        }}
        
        .token-tooltip::after {{
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 4px solid transparent;
            border-top-color: #333;
        }}
        
        .token-with-tooltip:hover .token-tooltip {{
            opacity: 1;
        }}
        
        /* Loading state */
        .loading {{
            text-align: center;
            padding: 50px;
            color: #666;
        }}
        
        .save-status {{
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
            text-align: center;
        }}
        
        .save-status.saved {{
            color: #2ecc71;
        }}
        
        .save-status.error {{
            color: #e74c3c;
        }}
        
        /* Context panel styles */
        .context-header {{
            position: sticky;
            top: 0;
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #ddd;
            z-index: 100;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        
        .context-header-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }}
        
        .context-header-left {{
            display: flex;
            flex-direction: column;
        }}
        
        .context-title {{
            font-size: 1.2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        
        .context-info {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .rollout-navigation {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .rollout-nav-button {{
            background: #fff;
            border: 1px solid #ddd;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1.1em;
            transition: all 0.2s;
        }}
        
        .rollout-nav-button:hover {{
            background: #f0f0f0;
            border-color: #999;
        }}
        
        .rollout-nav-button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        
        .rollout-input {{
            width: 80px;
            padding: 6px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            text-align: center;
            font-size: 1em;
        }}
        
        .rollout-input:focus {{
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        }}
        
        /* Highlight control sliders */
        .highlight-controls {{
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .slider-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .slider-label {{
            font-size: 0.85em;
            color: #666;
            min-width: 80px;
        }}
        
        .highlight-slider {{
            width: 120px;
            height: 6px;
            -webkit-appearance: none;
            appearance: none;
            background: #ddd;
            border-radius: 3px;
            outline: none;
        }}
        
        .highlight-slider.threshold-slider {{
            width: 180px;
        }}
        
        .highlight-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 16px;
            height: 16px;
            background: #3498db;
            border-radius: 50%;
            cursor: pointer;
        }}
        
        .highlight-slider::-moz-range-thumb {{
            width: 16px;
            height: 16px;
            background: #3498db;
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }}
        
        .slider-value {{
            font-size: 0.85em;
            color: #333;
            min-width: 40px;
            text-align: right;
        }}
        
        .context-content {{
            flex: 1;
            padding: 20px;
            padding-right: 40px; /* Extra padding for position indicator and scrollbar */
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.75em;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow-y: auto;
        }}
        
        .target-token {{
            background-color: rgba(255, 0, 0, 0.2);
            border: 2px solid red;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }}
        
        .context-loading {{
            text-align: center;
            padding: 50px;
            color: #666;
        }}
        
        /* Position indicator (minimap) */
        .position-indicator {{
            position: absolute;
            right: 15px; /* Leave room for scrollbar */
            top: 0;
            width: 16px;
            height: 100%;
            background: #f0f0f0;
            border-left: 1px solid #ddd;
            border-right: 1px solid #ddd;
        }}
        
        .position-marker {{
            position: absolute;
            left: 0;
            width: 100%;
            height: 3px;
            background: black;
            border: 1px solid white;
            z-index: 10;
            transition: top 0.3s ease;
        }}
        
        .activation-heatmap {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}
        
        .heatmap-line {{
            position: absolute;
            left: 0;
            width: 100%;
            opacity: 0.8;
        }}
        
        .no-feature-message {{
            text-align: center;
            padding: 50px;
            color: #666;
            font-size: 1.2em;
        }}
    </style>
</head>
<body>
    <div class="main-layout">
        <!-- Left Panel -->
        <div class="left-panel">
            <div class="container">
                <!-- Feature Selection Section -->
                <div class="feature-selection-section">
                    <div class="selection-title">Select Feature to Analyze</div>
                    <div class="selection-controls">
                        <div class="control-group">
                            <label class="control-label" for="layer-select">Layer:</label>
                            <select class="control-select" id="layer-select" onchange="updateProjectionOptions()">
                                <option value="">Select a layer...</option>
                            </select>
                        </div>
                        
                        <div class="control-group">
                            <label class="control-label" for="projection-select">Projection Type:</label>
                            <select class="control-select" id="projection-select" onchange="updatePolarityOptions()">
                                <option value="">Select projection type...</option>
                                <option value="gate_proj">gate_proj</option>
                                <option value="up_proj">up_proj</option>
                                <option value="down_proj">down_proj</option>
                            </select>
                        </div>
                        
                        <div class="control-group">
                            <label class="control-label" for="polarity-select">Polarity:</label>
                            <select class="control-select" id="polarity-select">
                                <option value="">Select polarity...</option>
                                <option value="positive">Positive</option>
                                <option value="negative">Negative</option>
                            </select>
                        </div>
                        
                        <button class="load-feature-button" onclick="loadSelectedFeature()">
                            Load Feature
                        </button>
                    </div>
                </div>
                
                <!-- Progress Section -->
                <div class="progress-section">
                    <div class="progress-title">Interpretation Progress</div>
                    <div class="progress-bar-container">
                        <div class="progress-bar" id="progress-bar" style="width: 0%">
                            <span id="progress-text">0%</span>
                        </div>
                    </div>
                    <div class="progress-stats">
                        <span id="interpreted-count">Interpreted: 0</span>
                        <span id="skipped-count">Skipped: 0</span>
                        <span id="remaining-count">Remaining: {total_features}</span>
                    </div>
                </div>
                
                <!-- Feature Display -->
                <div id="feature-container">
                    <div class="no-feature-message">Select a feature from the controls above to begin analysis</div>
                </div>
            </div>
        </div>
        
        <!-- Right Panel - Context Display -->
        <div class="right-panel" id="context-panel">
            <div class="context-header">
                <div class="context-header-top">
                    <div class="context-header-left">
                        <div class="context-title">Full Context</div>
                        <div class="context-info" id="context-info">Click on an example to view its full context</div>
                    </div>
                    <div class="rollout-navigation" id="rollout-navigation" style="display: none;">
                        <button class="rollout-nav-button" id="prev-rollout" onclick="navigateRollout(-1)">←</button>
                        <input type="number" class="rollout-input" id="rollout-input" placeholder="Rollout #" min="0">
                        <button class="rollout-nav-button" id="next-rollout" onclick="navigateRollout(1)">→</button>
                    </div>
                </div>
                <div class="highlight-controls">
                    <div class="slider-group">
                        <span class="slider-label">Threshold:</span>
                        <input type="range" class="highlight-slider threshold-slider" id="threshold-slider" min="0" max="10" step="0.05" value="0">
                        <span class="slider-value" id="threshold-value">0.00</span>
                    </div>
                    <div class="slider-group">
                        <span class="slider-label">Intensity:</span>
                        <input type="range" class="highlight-slider" id="intensity-slider" min="0.1" max="5" step="0.1" value="1">
                        <span class="slider-value" id="intensity-value">1.0x</span>
                    </div>
                </div>
            </div>
            <div class="context-wrapper">
                <div class="context-content" id="context-content">
                    <!-- Context will be loaded here -->
                </div>
                <div class="position-indicator" id="position-indicator">
                    <div class="activation-heatmap" id="activation-heatmap"></div>
                    <div class="position-marker" id="position-marker"></div>
                </div>
            </div>
        </div>
        
        <!-- Control Section with Interpretation -->
        <div class="control-section" id="control-section" style="display: none;">
            <div class="interpretation-mini">
                <textarea class="interpretation-mini-textarea" id="interpretation-text-mini" 
                          placeholder="Write your interpretation here..."></textarea>
                <div class="star-container-mini">
                    <input type="checkbox" class="star-checkbox" id="star-checkbox-mini">
                    <label for="star-checkbox-mini" class="star-label">⭐ Star</label>
                </div>
            </div>
            <div class="button-group">
                <button class="control-button save-button" onclick="saveFeature()">
                    Save Feature
                </button>
                <button class="control-button save-skip-button" onclick="saveSkip()">
                    Save Skip
                </button>
            </div>
        </div>
        
        <div class="save-status" id="save-status"></div>
    </div> <!-- end main-layout -->
    
    <script>
        // Store all features and current state
        const allFeatures = {json.dumps(all_features)};
        const totalFeatures = {total_features};
        let currentFeature = null;
        let interpretations = {{}};
        let contextCache = {{}}; // Cache loaded contexts
        let selectedExample = null;
        let activationsCache = {{}}; // Cache loaded activations
        let currentActivations = null; // Currently displayed activations
        let currentRolloutIdx = null; // Track current rollout index
        let currentTokenIdx = null; // Track current token index
        let maxRolloutIdx = null; // Track maximum rollout index
        let highlightThreshold = 0; // Minimum activation magnitude for highlighting
        let highlightIntensity = 1; // Multiplier for highlight intensity
        
        // API configuration
        const API_BASE = window.location.port === '8080' ? 'http://localhost:8085' : '';
        
        // Initialize layer options
        function initializeLayerOptions() {{
            const layerSelect = document.getElementById('layer-select');
            const layers = [...new Set(allFeatures.map(f => f.layer))].sort((a, b) => a - b);
            
            layers.forEach(layer => {{
                const option = document.createElement('option');
                option.value = layer;
                option.textContent = `Layer ${{layer}}`;
                layerSelect.appendChild(option);
            }});
        }}
        
        function updateProjectionOptions() {{
            const layerSelect = document.getElementById('layer-select');
            const projectionSelect = document.getElementById('projection-select');
            
            // Reset polarity when layer changes
            document.getElementById('polarity-select').value = '';
        }}
        
        function updatePolarityOptions() {{
            // No dynamic updates needed, but reset polarity selection
            document.getElementById('polarity-select').value = '';
        }}
        
        function loadSelectedFeature() {{
            const layer = parseInt(document.getElementById('layer-select').value);
            const projection = document.getElementById('projection-select').value;
            const polarity = document.getElementById('polarity-select').value;
            
            if (isNaN(layer) || !projection || !polarity) {{
                alert('Please select all options: layer, projection type, and polarity');
                return;
            }}
            
            // Find the matching feature
            const feature = allFeatures.find(f => 
                f.layer === layer && 
                f.projection === projection && 
                f.polarity === polarity
            );
            
            if (!feature) {{
                alert('Feature not found. Please check your selection.');
                return;
            }}
            
            currentFeature = feature;
            displayFeature(feature);
        }}
        
        async function loadInterpretations() {{
            try {{
                const response = await fetch(API_BASE + '/api/interpretations');
                if (response.ok) {{
                    const data = await response.json();
                    interpretations = data.interpretations || {{}};
                    updateProgress();
                }}
            }} catch (error) {{
                console.error('Failed to load interpretations:', error);
            }}
        }}
        
        function updateProgress() {{
            let interpreted = 0;
            let skipped = 0;
            
            Object.values(interpretations).forEach(interp => {{
                if (interp && typeof interp === 'object') {{
                    if (interp.skipped) {{
                        skipped++;
                    }} else if (interp.text && interp.text.trim()) {{
                        interpreted++;
                    }}
                }}
            }});
            
            const completed = interpreted + skipped;
            const remaining = totalFeatures - completed;
            const percentage = Math.round((completed / totalFeatures) * 100);
            
            document.getElementById('progress-bar').style.width = percentage + '%';
            document.getElementById('progress-text').textContent = percentage + '%';
            document.getElementById('interpreted-count').textContent = 'Interpreted: ' + interpreted;
            document.getElementById('skipped-count').textContent = 'Skipped: ' + skipped;
            document.getElementById('remaining-count').textContent = 'Remaining: ' + remaining;
        }}
        
        function displayFeature(feature) {{
            const container = document.getElementById('feature-container');
            const examples = feature.examples;
            
            let html = `
                <div class="feature-section">
                    <div class="feature-header">
                        <div class="feature-title">Layer ${{feature.layer}} - ${{feature.projection}} (${{feature.polarity}})</div>
                        <div class="feature-subtitle">Analyzing top activating examples</div>
                    </div>
                    <div class="examples-container">
            `;
            
            // Show all examples
            examples.forEach((example, idx) => {{
                const rolloutIdx = example.rollout_idx;
                const tokenIdx = example.token_idx;
                const activation = example.activation.toFixed(3);
                const tokenHtml = generateTokenHtml(example);
                const exampleNum = idx + 1;
                html += 
                    '<div class="example-item" onclick="selectExample(' + idx + ', ' + rolloutIdx + ', ' + tokenIdx + ')">' +
                        '<div class="example-info">Rollout ' + rolloutIdx + ', Example ' + exampleNum + ', Activation: ' + activation + '</div>' +
                        '<div>' + tokenHtml + '</div>' +
                    '</div>';
            }});
            
            html += `
                    </div>
                </div>
            `;
            
            container.innerHTML = html;
            document.getElementById('control-section').style.display = 'flex';
            
            // Load existing interpretation if any
            const existing = interpretations[feature.key];
            if (existing) {{
                document.getElementById('interpretation-text-mini').value = existing.text || '';
                document.getElementById('star-checkbox-mini').checked = existing.starred || false;
            }} else {{
                document.getElementById('interpretation-text-mini').value = '';
                document.getElementById('star-checkbox-mini').checked = false;
            }}
        }}
        
        function generateTokenHtml(example) {{
            const tokens = example.context;
            const activations = example.context_activations;
            const targetIdx = example.target_position;
            
            let html = '';
            tokens.forEach((token, i) => {{
                const activation = activations[i];
                const intensity = Math.min(Math.abs(activation) * 0.1, 0.7);
                const bgColor = activation > 0 
                    ? 'rgba(255, 0, 0, ' + intensity + ')' 
                    : 'rgba(0, 0, 255, ' + intensity + ')';
                
                const tokenDisplay = token.replace(/\\n/g, '\\\\n').replace(/ /g, '&nbsp;');
                
                if (i === targetIdx) {{
                    html += '<span class="token-with-tooltip" style="background-color: ' + bgColor + '; border: 2px solid red; font-weight: bold; padding: 2px 1px; border-radius: 2px; position: relative; display: inline-block;">';
                }} else {{
                    html += '<span class="token-with-tooltip" style="background-color: ' + bgColor + '; padding: 2px 1px; border-radius: 2px; position: relative; display: inline-block;">';
                }}
                
                const activationStr = activation.toFixed(3);
                html += tokenDisplay + '<span class="token-tooltip">' + activationStr + '</span></span>';
            }});
            
            return html;
        }}
        
        async function saveInterpretation(skipFeature = false) {{
            if (!currentFeature) {{
                alert('No feature loaded. Please select a feature first.');
                return;
            }}
            
            const text = document.getElementById('interpretation-text-mini').value;
            const starred = document.getElementById('star-checkbox-mini').checked;
            
            const statusEl = document.getElementById('save-status');
            statusEl.textContent = 'Saving...';
            statusEl.className = 'save-status';
            
            try {{
                const response = await fetch(API_BASE + '/api/interpretations', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        featureKey: currentFeature.key,
                        text: text,
                        starred: starred,
                        skipped: skipFeature
                    }})
                }});
                
                if (response.ok) {{
                    const data = await response.json();
                    if (data && data.interpretation) {{
                        interpretations[currentFeature.key] = data.interpretation;
                    }} else {{
                        // Create a minimal interpretation object if the API doesn't return one
                        interpretations[currentFeature.key] = {{
                            text: text,
                            starred: starred,
                            skipped: skipFeature,
                            lastModified: new Date().toISOString()
                        }};
                    }}
                    
                    statusEl.textContent = 'Saved!';
                    statusEl.className = 'save-status saved';
                    
                    updateProgress();
                    
                    setTimeout(() => {{
                        statusEl.textContent = '';
                    }}, 2000);
                }} else {{
                    throw new Error('Save failed');
                }}
            }} catch (error) {{
                console.error('Failed to save:', error);
                statusEl.textContent = 'Error saving';
                statusEl.className = 'save-status error';
            }}
        }}
        
        function saveFeature() {{
            saveInterpretation(false);
        }}
        
        function saveSkip() {{
            saveInterpretation(true);
        }}
        
        async function loadRolloutContext(rolloutIdx, tokenIdx, fromNavigation = false) {{
            const contextPanel = document.getElementById('context-panel');
            const contextContent = document.getElementById('context-content');
            const contextInfo = document.getElementById('context-info');
            const rolloutNav = document.getElementById('rollout-navigation');
            const rolloutInput = document.getElementById('rollout-input');
            
            // Update current rollout and token indices
            currentRolloutIdx = rolloutIdx;
            currentTokenIdx = tokenIdx;
            
            // Show the context panel
            contextPanel.style.display = 'flex';
            
            // Show navigation controls and update input
            rolloutNav.style.display = 'flex';
            rolloutInput.value = rolloutIdx;
            
            // Show loading state
            contextContent.innerHTML = '<div class="context-loading">Loading context and activations...</div>';
            contextInfo.textContent = 'Rollout ' + rolloutIdx;
            
            // If navigating by rollout number, use token 0 as default
            if (fromNavigation && tokenIdx === null) {{
                tokenIdx = 0;
                currentTokenIdx = 0;
            }}
            
            try {{
                // Load context and activations in parallel
                const [contextData, activations] = await Promise.all([
                    // Load context if not cached
                    contextCache[rolloutIdx] || fetch(API_BASE + '/api/rollout_context/' + rolloutIdx).then(r => r.json()),
                    // Load activations
                    loadActivations(rolloutIdx)
                ]);
                
                // Cache context if it was just loaded
                if (!contextCache[rolloutIdx]) {{
                    contextCache[rolloutIdx] = contextData;
                }}
                
                // Store current activations
                currentActivations = activations;
                
                // Display with activations
                displayContext(contextData.text, contextData.tokens, tokenIdx, activations);
                
                // Update navigation button states
                updateNavigationButtons();
            }} catch (error) {{
                console.error('Failed to load context/activations:', error);
                contextContent.innerHTML = '<div class="context-loading">Error loading data</div>';
            }}
        }}
        
        function navigateRollout(direction) {{
            if (currentRolloutIdx === null) return;
            
            const newIdx = currentRolloutIdx + direction;
            if (newIdx >= 0 && (maxRolloutIdx === null || newIdx <= maxRolloutIdx)) {{
                loadRolloutContext(newIdx, null, true);
            }}
        }}
        
        function updateNavigationButtons() {{
            const prevButton = document.getElementById('prev-rollout');
            const nextButton = document.getElementById('next-rollout');
            
            if (currentRolloutIdx !== null) {{
                prevButton.disabled = currentRolloutIdx <= 0;
                nextButton.disabled = maxRolloutIdx !== null && currentRolloutIdx >= maxRolloutIdx;
            }}
        }}
        
        function refreshContextDisplay() {{
            // Re-display current context with updated highlight settings
            if (currentRolloutIdx !== null && contextCache[currentRolloutIdx]) {{
                const contextData = contextCache[currentRolloutIdx];
                displayContext(contextData.text, contextData.tokens, currentTokenIdx, currentActivations, true);
            }}
        }}
        
        function displayContext(fullText, tokens, tokenIdx, activations, fromSliderUpdate = false) {{
            const contextContent = document.getElementById('context-content');
            
            if (!tokens || tokens.length === 0) {{
                // Fallback: just display the text without highlighting
                const escapedText = fullText
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#039;');
                contextContent.innerHTML = escapedText;
                return;
            }}
            
            // Get activation for current feature if available
            let tokenActivations = null;
            console.log('displayContext - currentFeature:', currentFeature);
            console.log('displayContext - activations:', activations);
            
            if (activations && currentFeature) {{
                const layerIdx = currentFeature.layer;
                const projIdx = ['gate_proj', 'up_proj', 'down_proj'].indexOf(currentFeature.projection);
                const [numTokens, numLayers, numProj] = activations.shape;
                
                console.log('Extracting activations for layer', layerIdx, 'projection', currentFeature.projection, 'projIdx', projIdx);
                console.log('Shape:', numTokens, 'tokens,', numLayers, 'layers,', numProj, 'projections');
                
                // Find layer position in the data
                // The activations are stored for all layers in order
                let layerPos = layerIdx; // Direct mapping since layers start from 0
                
                if (layerPos >= 0 && layerPos < numLayers && projIdx >= 0) {{
                    // Extract activations for this feature
                    tokenActivations = new Float32Array(numTokens);
                    for (let t = 0; t < numTokens; t++) {{
                        const idx = t * numLayers * numProj + layerPos * numProj + projIdx;
                        tokenActivations[t] = activations.data[idx];
                    }}
                    console.log('Extracted activations, first few values:', tokenActivations.slice(0, 5));
                }} else {{
                    console.log('Invalid layer position or projection index');
                }}
            }} else {{
                console.log('Missing activations or currentFeature');
            }}
            
            // Build the text with highlighted token and activation overlays
            let html = '';
            
            // Concatenate tokens to rebuild the text with highlighting
            tokens.forEach((token, idx) => {{
                // Escape the token
                let escapedToken = token
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#039;');
                
                // Check if token contains newline and handle specially
                let displayToken = escapedToken;
                let hasNewline = token.includes('\\n');
                if (hasNewline) {{
                    // First, replace all newlines with visible \\n
                    let visibleNewlines = escapedToken.replace(/\\n/g, '<span style="opacity: 0.5;">\\\\n</span>');
                    // Then add line breaks for each original newline
                    const newlineCount = (token.match(/\\n/g) || []).length;
                    displayToken = visibleNewlines + '<br>'.repeat(newlineCount);
                }}
                
                // Calculate activation background if available
                let style = '';
                if (tokenActivations && idx < tokenActivations.length) {{
                    const activation = tokenActivations[idx];
                    const polarity = currentFeature.polarity;
                    
                    // Only show activation if it matches the polarity we're looking at
                    if ((polarity === 'positive' && activation > 0) || 
                        (polarity === 'negative' && activation < 0)) {{
                        const absActivation = Math.abs(activation);
                        // Apply threshold and intensity multiplier
                        if (absActivation >= highlightThreshold) {{
                            const intensity = Math.min(absActivation * 0.1 * highlightIntensity, 0.9);
                            const color = polarity === 'positive' 
                                ? 'rgba(255, 0, 0, ' + intensity + ')' 
                                : 'rgba(0, 0, 255, ' + intensity + ')';
                            style = 'style="background-color: ' + color + ';"';
                        }}
                    }}
                }}
                
                if (idx === tokenIdx) {{
                    // Highlight the target token with border
                    html += '<span class="target-token" id="target-token" ' + style + '>' + displayToken + '</span>';
                }} else {{
                    // Regular token with activation background
                    if (style) {{
                        html += '<span ' + style + '>' + displayToken + '</span>';
                    }} else {{
                        html += displayToken;
                    }}
                }}
            }});
            
            contextContent.innerHTML = html;
            
            // Build activation heatmap
            if (tokenActivations && currentFeature) {{
                buildActivationHeatmap(tokens, tokenActivations);
            }}
            
            // Scroll to the highlighted token only if not from a slider update
            if (!fromSliderUpdate) {{
                setTimeout(() => {{
                    const targetElement = document.getElementById('target-token');
                    if (targetElement) {{
                        targetElement.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                        updatePositionMarker();
                    }}
                }}, 100);
            }}
        }}
        
        function buildActivationHeatmap(tokens, tokenActivations) {{
            // Wait a bit for DOM to settle
            setTimeout(() => {{
                const heatmapContainer = document.getElementById('activation-heatmap');
                const contextContent = document.getElementById('context-content');
                const polarity = currentFeature.polarity;
                
                if (!heatmapContainer || !contextContent) return;
                
                // Clear existing heatmap
                heatmapContainer.innerHTML = '';
                
                // Get all spans in the content
                const allSpans = contextContent.querySelectorAll('span');
                const contentRect = contextContent.getBoundingClientRect();
                const scrollTop = contextContent.scrollTop;
                
                // Group spans by line position
                const lineMap = new Map(); // line position -> activations array
                
                allSpans.forEach((span, idx) => {{
                    if (idx >= tokenActivations.length) return;
                    
                    const rect = span.getBoundingClientRect();
                    const relativeTop = rect.top - contentRect.top + scrollTop;
                    const lineKey = Math.floor(relativeTop / 20); // Group by ~20px lines
                    
                    if (!lineMap.has(lineKey)) {{
                        lineMap.set(lineKey, []);
                    }}
                    lineMap.get(lineKey).push(tokenActivations[idx]);
                }});
                
                // Also process text nodes that aren't in spans
                let tokenIdx = 0;
                for (let node of contextContent.childNodes) {{
                    if (node.nodeType === Node.TEXT_NODE && node.textContent.trim() && tokenIdx < tokenActivations.length) {{
                        // This is a token without activation styling
                        const range = document.createRange();
                        range.selectNode(node);
                        const rect = range.getBoundingClientRect();
                        const relativeTop = rect.top - contentRect.top + scrollTop;
                        const lineKey = Math.floor(relativeTop / 20);
                        
                        if (!lineMap.has(lineKey)) {{
                            lineMap.set(lineKey, []);
                        }}
                        lineMap.get(lineKey).push(tokenActivations[tokenIdx]);
                        tokenIdx++;
                    }} else if (node.nodeType === Node.ELEMENT_NODE) {{
                        tokenIdx++;
                    }}
                }}
                
                // Create heatmap lines
                const contentHeight = contextContent.scrollHeight;
                
                lineMap.forEach((activations, lineKey) => {{
                    // Find max activation matching polarity
                    let maxActivation = 0;
                    activations.forEach(activation => {{
                        if ((polarity === 'positive' && activation > 0) || 
                            (polarity === 'negative' && activation < 0)) {{
                            maxActivation = Math.max(maxActivation, Math.abs(activation));
                        }}
                    }});
                    
                    if (maxActivation > 0 && maxActivation >= highlightThreshold) {{
                        const lineTop = (lineKey * 20 / contentHeight) * 100;
                        const lineHeight = (20 / contentHeight) * 100;
                        
                        const heatmapLine = document.createElement('div');
                        heatmapLine.className = 'heatmap-line';
                        heatmapLine.style.top = lineTop + '%';
                        heatmapLine.style.height = Math.max(lineHeight, 0.5) + '%'; // Min 0.5% height
                        
                        // Color based on intensity with multiplier
                        const intensity = Math.min(maxActivation * 0.15 * highlightIntensity, 0.9);
                        const color = polarity === 'positive' 
                            ? 'rgba(255, 0, 0, ' + intensity + ')' 
                            : 'rgba(0, 0, 255, ' + intensity + ')';
                        heatmapLine.style.backgroundColor = color;
                        
                        heatmapContainer.appendChild(heatmapLine);
                    }}
                }});
            }}, 150); // Delay to ensure DOM is rendered
        }}
        
        function updatePositionMarker() {{
            const targetElement = document.getElementById('target-token');
            const contextContent = document.getElementById('context-content');
            const positionMarker = document.getElementById('position-marker');
            
            if (!targetElement || !contextContent || !positionMarker) return;
            
            // Calculate the position of the target token relative to the content
            const contentHeight = contextContent.scrollHeight;
            const tokenOffset = targetElement.offsetTop - contextContent.offsetTop;
            const markerPosition = (tokenOffset / contentHeight) * 100;
            
            // Update the marker position
            positionMarker.style.top = markerPosition + '%';
        }}
        
        function selectExample(exampleIdx, rolloutIdx, tokenIdx) {{
            // Update selected state
            const allExamples = document.querySelectorAll('.example-item');
            allExamples.forEach((el, idx) => {{
                if (idx === exampleIdx) {{
                    el.classList.add('selected');
                }} else {{
                    el.classList.remove('selected');
                }}
            }});
            
            // Load the context
            selectedExample = exampleIdx;
            loadRolloutContext(rolloutIdx, tokenIdx, false);  // false indicates this is from clicking an example
        }}
        
        // Initialize on load
        window.addEventListener('DOMContentLoaded', async () => {{
            initializeLayerOptions();
            await loadInterpretations();
            
            // Initialize highlight control sliders
            const thresholdSlider = document.getElementById('threshold-slider');
            const thresholdValue = document.getElementById('threshold-value');
            const intensitySlider = document.getElementById('intensity-slider');
            const intensityValue = document.getElementById('intensity-value');
            
            if (thresholdSlider && thresholdValue) {{
                thresholdSlider.addEventListener('input', (e) => {{
                    highlightThreshold = parseFloat(e.target.value);
                    thresholdValue.textContent = highlightThreshold.toFixed(2);
                    // Refresh current display if context is loaded
                    if (currentRolloutIdx !== null) {{
                        refreshContextDisplay();
                    }}
                }});
            }}
            
            if (intensitySlider && intensityValue) {{
                intensitySlider.addEventListener('input', (e) => {{
                    highlightIntensity = parseFloat(e.target.value);
                    intensityValue.textContent = highlightIntensity.toFixed(1) + 'x';
                    // Refresh current display if context is loaded
                    if (currentRolloutIdx !== null) {{
                        refreshContextDisplay();
                    }}
                }});
            }}
            
            // Add event listener for rollout input
            const rolloutInput = document.getElementById('rollout-input');
            if (rolloutInput) {{
                rolloutInput.addEventListener('keypress', (e) => {{
                    if (e.key === 'Enter') {{
                        const rolloutIdx = parseInt(rolloutInput.value);
                        if (!isNaN(rolloutIdx) && rolloutIdx >= 0) {{
                            loadRolloutContext(rolloutIdx, null, true);
                        }}
                    }}
                }});
                
                rolloutInput.addEventListener('blur', () => {{
                    const rolloutIdx = parseInt(rolloutInput.value);
                    if (!isNaN(rolloutIdx) && rolloutIdx >= 0) {{
                        loadRolloutContext(rolloutIdx, null, true);
                    }}
                }});
            }}
            
            // Extract max rollout index from data if available
            if (typeof allFeatures !== 'undefined' && allFeatures.length > 0) {{
                maxRolloutIdx = 0;
                allFeatures.forEach(feature => {{
                    feature.examples.forEach(example => {{
                        if (example.rollout_idx > maxRolloutIdx) {{
                            maxRolloutIdx = example.rollout_idx;
                        }}
                    }});
                }});
            }}
            
            // Add scroll listener for context panel
            const contextContent = document.getElementById('context-content');
            if (contextContent) {{
                contextContent.addEventListener('scroll', () => {{
                    updateScrollIndicator();
                    // Rebuild heatmap on scroll if we have activations
                    if (currentActivations && currentFeature) {{
                        const tokens = contextCache[currentActivations.rolloutIdx]?.tokens;
                        if (tokens) {{
                            // Extract activations for current feature
                            const layerIdx = currentFeature.layer;
                            const projIdx = ['gate_proj', 'up_proj', 'down_proj'].indexOf(currentFeature.projection);
                            const [numTokens, numLayers, numProj] = currentActivations.shape;
                            
                            let tokenActivations = null;
                            // Direct mapping since layers are stored in order (0-63)
                            let layerPos = layerIdx;
                            
                            if (layerPos >= 0 && layerPos < numLayers && projIdx >= 0) {{
                                tokenActivations = new Float32Array(numTokens);
                                for (let t = 0; t < numTokens; t++) {{
                                    const idx = t * numLayers * numProj + layerPos * numProj + projIdx;
                                    tokenActivations[t] = currentActivations.data[idx];
                                }}
                                buildActivationHeatmap(tokens, tokenActivations);
                            }}
                        }}
                    }}
                }});
            }}
        }});
        
        function updateScrollIndicator() {{
            const contextContent = document.getElementById('context-content');
            const positionIndicator = document.getElementById('position-indicator');
            
            if (!contextContent || !positionIndicator) return;
            
            // You could add a viewport indicator here if desired
            // For now, we just ensure the marker stays visible
        }}
        
        async function loadActivations(rolloutIdx) {{
            // Check cache first
            if (activationsCache[rolloutIdx]) {{
                return activationsCache[rolloutIdx];
            }}
            
            try {{
                const response = await fetch(API_BASE + '/api/activations/' + rolloutIdx);
                if (!response.ok) {{
                    throw new Error('Failed to load activations');
                }}
                
                const data = await response.json();
                
                // Decode base64
                const binaryString = atob(data.data);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {{
                    bytes[i] = binaryString.charCodeAt(i);
                }}
                
                // Decompress using pako (we'll need to include this library)
                const decompressed = pako.inflate(bytes);
                
                // Convert to Float32Array (JS doesn't have Float16)
                const float16Buffer = new ArrayBuffer(decompressed.length);
                const float16View = new Uint8Array(float16Buffer);
                float16View.set(decompressed);
                
                // For now, treat as Float32 (we'll lose some precision)
                const numFloats = decompressed.length / 2;
                const floatArray = new Float32Array(numFloats);
                const dataView = new DataView(float16Buffer);
                
                // Simple float16 to float32 conversion
                for (let i = 0; i < numFloats; i++) {{
                    const float16 = dataView.getUint16(i * 2, true);
                    // Simplified conversion - proper float16 conversion would be more complex
                    const sign = (float16 >> 15) & 1;
                    const exponent = (float16 >> 10) & 0x1f;
                    const fraction = float16 & 0x3ff;
                    
                    if (exponent === 0) {{
                        floatArray[i] = (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
                    }} else if (exponent === 31) {{
                        floatArray[i] = fraction ? NaN : (sign ? -Infinity : Infinity);
                    }} else {{
                        floatArray[i] = (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
                    }}
                }}
                
                // Reshape to [num_tokens, num_layers, 3]
                const activations = {{
                    data: floatArray,
                    shape: data.shape,
                    rolloutIdx: rolloutIdx
                }};
                
                // Cache it (limit cache size to 10 rollouts)
                const cacheKeys = Object.keys(activationsCache);
                if (cacheKeys.length >= 10) {{
                    delete activationsCache[cacheKeys[0]];
                }}
                activationsCache[rolloutIdx] = activations;
                
                return activations;
            }} catch (error) {{
                console.error('Failed to load activations:', error);
                return null;
            }}
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.ctrlKey || e.metaKey) {{
                if (e.key === 'Enter') {{
                    e.preventDefault();
                    saveFeature();
                }} else if (e.key === 's') {{
                    e.preventDefault();
                    saveSkip();
                }}
            }}
        }});
    </script>
</body>
</html>"""
    
    # Format the HTML with data
    html_content = html_content.replace('{features_json}', json.dumps(all_features))
    html_content = html_content.replace('{total_features}', str(total_features))
    
    # Write to file
    print(f"Writing dashboard to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Dashboard generated successfully!")
    print(f"Total features: {total_features}")
    print(f"Open {output_path} in your browser to start analyzing specific features.")


def main():
    parser = argparse.ArgumentParser(description="Generate feature selection dashboard")
    parser.add_argument("--data", default="backend/activations_data.json", 
                       help="Path to activation data JSON file")
    parser.add_argument("--output", default="feature_selection_dashboard.html",
                       help="Output HTML file path")
    
    args = parser.parse_args()
    
    # Find the data file
    if not os.path.exists(args.data):
        possible_paths = [
            "backend/activations_data.json",
            "activations_data.json"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                args.data = path
                break
        else:
            print(f"Error: Could not find activation data file at {args.data}")
            return 1
    
    generate_dashboard_html(args.data, args.output)
    return 0


if __name__ == "__main__":
    exit(main())