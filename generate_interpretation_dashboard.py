#!/usr/bin/env python3
"""
Generate a streamlined HTML dashboard for interpreting LoRA features one at a time.
Shows a single feature with no layer/projection information to reduce bias.
"""

import json
import os
import argparse
import html as html_lib
from datetime import datetime
import random


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
    """Generate the interpretation-focused dashboard"""
    
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
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <title>LoRA Feature Interpretation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 0;
            margin: 0;
            overflow: hidden;
        }
        
        .main-layout {
            display: flex;
            height: 100vh;
            width: 100vw;
        }
        
        .left-panel {
            flex: 0 0 60%;
            padding: 20px;
            overflow-y: auto;
            padding-bottom: 100px; /* Space for fixed controls */
        }
        
        .right-panel {
            flex: 0 0 40%;
            background: white;
            border-left: 1px solid #ddd;
            position: relative;
            display: none; /* Hidden initially, will be changed to flex when shown */
            flex-direction: column;
        }
        
        .context-wrapper {
            position: relative;
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        
        .container {
            max-width: 100%;
            margin: 0 auto;
        }
        
        /* Progress section */
        .progress-section {
            background: white;
            padding: 10px 15px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        
        .progress-title {
            font-size: 0.9em;
            font-weight: bold;
            margin-bottom: 6px;
            color: #2c3e50;
        }
        
        .progress-bar-container {
            background: #e0e0e0;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(to right, #3498db, #2ecc71);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .progress-stats {
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
            font-size: 0.9em;
            color: #666;
        }
        
        /* Feature display */
        .feature-section {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .feature-header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .feature-title {
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .feature-subtitle {
            color: #666;
            font-size: 1.1em;
        }
        
        /* Examples */
        .examples-container {
            margin-bottom: 30px;
        }
        
        .example-item {
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
        }
        
        .example-item:hover {
            background: #e9ecef;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .example-item.selected {
            background: #d4edda;
            border: 2px solid #28a745;
        }
        
        .example-info {
            font-size: 0.85em;
            color: #666;
            margin-bottom: 8px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        
        /* Control section */
        .control-section {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            padding: 15px 20px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 30px;
            z-index: 1000;
        }
        
        .interpretation-mini {
            display: flex;
            align-items: center;
            gap: 15px;
            flex: 0 1 600px;
        }
        
        .interpretation-mini-textarea {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.95em;
            resize: none;
            height: 50px;
            background: white;
        }
        
        .interpretation-mini-textarea:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        }
        
        .star-container-mini {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .button-group {
            display: flex;
            gap: 15px;
        }
        
        .control-button {
            padding: 12px 30px;
            font-size: 1.1em;
            font-weight: bold;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .next-button {
            background: #2ecc71;
            color: white;
        }
        
        .next-button:hover {
            background: #27ae60;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .skip-button {
            background: #e74c3c;
            color: white;
        }
        
        .skip-button:hover {
            background: #c0392b;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Completion message */
        .completion-message {
            text-align: center;
            padding: 50px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .completion-title {
            font-size: 2em;
            color: #2ecc71;
            margin-bottom: 20px;
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
        
        /* Loading state */
        .loading {
            text-align: center;
            padding: 50px;
            color: #666;
        }
        
        .save-status {
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
            text-align: center;
        }
        
        .save-status.saved {
            color: #2ecc71;
        }
        
        .save-status.error {
            color: #e74c3c;
        }
        
        /* Context panel styles */
        .context-header {
            position: sticky;
            top: 0;
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #ddd;
            z-index: 100;
        }
        
        .context-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        
        .context-info {
            font-size: 0.9em;
            color: #666;
        }
        
        .context-content {
            flex: 1;
            padding: 20px;
            padding-right: 40px; /* Extra padding for position indicator and scrollbar */
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.75em;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow-y: auto;
        }
        
        .target-token {
            background-color: rgba(255, 0, 0, 0.2);
            border: 2px solid red;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }
        
        .context-loading {
            text-align: center;
            padding: 50px;
            color: #666;
        }
        
        /* Position indicator (minimap) */
        .position-indicator {
            position: absolute;
            right: 15px; /* Leave room for scrollbar */
            top: 0;
            width: 12px;
            height: 100%;
            background: #f0f0f0;
            border-left: 1px solid #ddd;
            border-right: 1px solid #ddd;
        }
        
        .position-marker {
            position: absolute;
            left: 0;
            width: 100%;
            height: 3px;
            background: red;
            transition: top 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="main-layout">
        <!-- Left Panel -->
        <div class="left-panel">
            <div class="container">
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
                    <div class="loading">Loading features...</div>
                </div>
            </div>
        </div>
        
        <!-- Right Panel - Context Display -->
        <div class="right-panel" id="context-panel">
            <div class="context-header">
                <div class="context-title">Full Context</div>
                <div class="context-info" id="context-info">Click on an example to view its full context</div>
            </div>
            <div class="context-wrapper">
                <div class="context-content" id="context-content">
                    <!-- Context will be loaded here -->
                </div>
                <div class="position-indicator" id="position-indicator">
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
                <button class="control-button next-button" onclick="nextFeature()">
                    Next Feature →
                </button>
                <button class="control-button skip-button" onclick="skipFeature()">
                    Skip This Feature
                </button>
            </div>
        </div>
        
        <div class="save-status" id="save-status"></div>
    </div> <!-- end main-layout -->
    
    <script>
        // Store all features and current state
        const allFeatures = {features_json};
        const totalFeatures = {total_features};
        let currentFeature = null;
        let interpretations = {};
        let contextCache = {}; // Cache loaded contexts
        let selectedExample = null;
        
        // API configuration
        const API_BASE = window.location.port === '8080' ? 'http://localhost:8085' : '';
        
        async function loadInterpretations() {{
            try {{
                const response = await fetch(API_BASE + '/api/interpretations');
                if (response.ok) {{
                    const data = await response.json();
                    interpretations = data.interpretations || {};
                    updateProgress();
                    loadNextFeature();
                }}
            }} catch (error) {{
                console.error('Failed to load interpretations:', error);
                loadNextFeature();
            }}
        }}
        
        function updateProgress() {{
            let interpreted = 0;
            let skipped = 0;
            
            Object.values(interpretations).forEach(interp => {{
                if (interp.skipped) {{
                    skipped++;
                }} else if (interp.text && interp.text.trim()) {{
                    interpreted++;
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
        
        function getUnannotatedFeatures() {{
            return allFeatures.filter(feature => {{
                const interp = interpretations[feature.key];
                return !interp || (!interp.text && !interp.skipped);
            }});
        }}
        
        function loadNextFeature() {{
            const unannotated = getUnannotatedFeatures();
            
            if (unannotated.length === 0) {{
                showCompletionMessage();
                return;
            }}
            
            // Random selection
            const randomIndex = Math.floor(Math.random() * unannotated.length);
            currentFeature = unannotated[randomIndex];
            displayFeature(currentFeature);
        }}
        
        function displayFeature(feature) {{
            const container = document.getElementById('feature-container');
            const examples = feature.examples;
            
            let html = `
                <div class="feature-section">
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
            if (!currentFeature) return;
            
            const text = document.getElementById('interpretation-text-mini').value;
            const starred = document.getElementById('star-checkbox-mini').checked;
            
            const statusEl = document.getElementById('save-status');
            statusEl.textContent = 'Saving...';
            statusEl.className = 'save-status';
            
            try {{
                const response = await fetch(API_BASE + '/api/interpretations', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        featureKey: currentFeature.key,
                        text: text,
                        starred: starred,
                        skipped: skipFeature
                    })
                });
                
                if (response.ok) {{
                    const data = await response.json();
                    interpretations[currentFeature.key] = data.interpretation;
                    
                    statusEl.textContent = 'Saved!';
                    statusEl.className = 'save-status saved';
                    
                    updateProgress();
                    
                    setTimeout(() => {{
                        statusEl.textContent = '';
                        loadNextFeature();
                    }}, 500);
                }} else {{
                    throw new Error('Save failed');
                }}
            }} catch (error) {{
                console.error('Failed to save:', error);
                statusEl.textContent = 'Error saving';
                statusEl.className = 'save-status error';
            }}
        }}
        
        function nextFeature() {{
            saveInterpretation(false);
        }}
        
        function skipFeature() {{
            saveInterpretation(true);
        }}
        
        function showCompletionMessage() {{
            const container = document.getElementById('feature-container');
            container.innerHTML = 
                '<div class="completion-message">' +
                    '<div class="completion-title">🎉 All Features Reviewed!</div>' +
                    '<p>You\\'ve gone through all available features.</p>' +
                    '<p>Total features: ' + totalFeatures + '</p>' +
                '</div>';
            document.getElementById('control-section').style.display = 'none';
        }}
        
        async function loadRolloutContext(rolloutIdx, tokenIdx) {{
            const contextPanel = document.getElementById('context-panel');
            const contextContent = document.getElementById('context-content');
            const contextInfo = document.getElementById('context-info');
            
            // Show the context panel
            contextPanel.style.display = 'flex';
            
            // Check cache first
            if (contextCache[rolloutIdx]) {{
                displayContext(contextCache[rolloutIdx].text, contextCache[rolloutIdx].tokens, tokenIdx);
                return;
            }}
            
            // Show loading state
            contextContent.innerHTML = '<div class="context-loading">Loading context...</div>';
            contextInfo.textContent = 'Rollout ' + rolloutIdx;
            
            try {{
                const response = await fetch(API_BASE + '/api/rollout_context/' + rolloutIdx);
                if (response.ok) {{
                    const data = await response.json();
                    contextCache[rolloutIdx] = data;
                    displayContext(data.text, data.tokens, tokenIdx);
                }} else {{
                    contextContent.innerHTML = '<div class="context-loading">Failed to load context</div>';
                }}
            }} catch (error) {{
                console.error('Failed to load context:', error);
                contextContent.innerHTML = '<div class="context-loading">Error loading context</div>';
            }}
        }}
        
        function displayContext(fullText, tokens, tokenIdx) {{
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
            
            // Build the text with highlighted token
            let html = '';
            let currentPos = 0;
            
            // Concatenate tokens to rebuild the text with highlighting
            tokens.forEach((token, idx) => {{
                // Escape the token
                const escapedToken = token
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#039;');
                
                if (idx === tokenIdx) {{
                    // Highlight the target token
                    html += '<span class="target-token" id="target-token">' + escapedToken + '</span>';
                }} else {{
                    html += escapedToken;
                }}
            }});
            
            contextContent.innerHTML = html;
            
            // Scroll to the highlighted token
            setTimeout(() => {{
                const targetElement = document.getElementById('target-token');
                if (targetElement) {{
                    targetElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    updatePositionMarker();
                }}
            }}, 100);
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
            loadRolloutContext(rolloutIdx, tokenIdx);
        }}
        
        // Initialize on load
        window.addEventListener('DOMContentLoaded', async () => {{
            await loadInterpretations();
            
            // Add scroll listener for context panel
            const contextContent = document.getElementById('context-content');
            if (contextContent) {{
                contextContent.addEventListener('scroll', () => {{
                    updateScrollIndicator();
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
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.ctrlKey || e.metaKey) {{
                if (e.key === 'Enter') {{
                    e.preventDefault();
                    nextFeature();
                }} else if (e.key === 's') {{
                    e.preventDefault();
                    skipFeature();
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
    print(f"Open {output_path} in your browser to start interpreting features.")


def main():
    parser = argparse.ArgumentParser(description="Generate interpretation-focused dashboard")
    parser.add_argument("--data", default="backend/activations_data.json", 
                       help="Path to activation data JSON file")
    parser.add_argument("--output", default="interpretation_dashboard.html",
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