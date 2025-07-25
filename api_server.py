#!/usr/bin/env python3
"""
Standalone API server for interpretations.
Run this alongside python -m http.server
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
from datetime import datetime
import h5py
import numpy as np
import base64
import gzip

class APIHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        if self.path == '/api/interpretations':
            interpretations = {'interpretations': {}}
            if os.path.exists('interpretations.json'):
                with open('interpretations.json', 'r') as f:
                    interpretations = json.load(f)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(interpretations).encode())
        elif self.path.startswith('/api/rollout_context/'):
            # Extract rollout_idx from path
            try:
                rollout_idx = self.path.split('/')[-1]
                
                # Load rollout contexts
                contexts_path = 'backend/rollout_contexts.json'
                if not os.path.exists(contexts_path):
                    contexts_path = 'rollout_contexts.json'
                
                tokens_path = 'backend/rollout_tokens.json'
                if not os.path.exists(tokens_path):
                    tokens_path = 'rollout_tokens.json'
                
                if os.path.exists(contexts_path):
                    with open(contexts_path, 'r') as f:
                        contexts = json.load(f)
                    
                    # Load tokens if available
                    tokens = {}
                    if os.path.exists(tokens_path):
                        with open(tokens_path, 'r') as f:
                            tokens = json.load(f)
                    
                    if rollout_idx in contexts:
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        response = {
                            'rollout_idx': rollout_idx,
                            'text': contexts[rollout_idx],
                            'tokens': tokens.get(rollout_idx, [])
                        }
                        self.wfile.write(json.dumps(response).encode())
                    else:
                        self.send_error(404, f"Rollout {rollout_idx} not found")
                else:
                    self.send_error(404, "Rollout contexts file not found")
            except Exception as e:
                self.send_error(500, str(e))
        elif self.path.startswith('/api/activations/'):
            # Extract rollout_idx from path
            try:
                rollout_idx = self.path.split('/')[-1]
                
                # Find the HDF5 file
                h5_path = f'backend/activations/rollout_{rollout_idx}.h5'
                if not os.path.exists(h5_path):
                    h5_path = f'activations/rollout_{rollout_idx}.h5'
                
                if os.path.exists(h5_path):
                    # Load activations from HDF5
                    with h5py.File(h5_path, 'r') as f:
                        activations = f['activations'][:]
                        shape = list(activations.shape)
                    
                    # Convert to float16 and compress
                    activations_f16 = activations.astype(np.float16)
                    compressed = gzip.compress(activations_f16.tobytes(), compresslevel=1)
                    encoded = base64.b64encode(compressed).decode('ascii')
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    
                    response = {
                        'rollout_idx': rollout_idx,
                        'shape': shape,
                        'dtype': 'float16',
                        'data': encoded
                    }
                    self.wfile.write(json.dumps(response).encode())
                else:
                    self.send_error(404, f"Activations for rollout {rollout_idx} not found")
            except Exception as e:
                print(f"Error serving activations: {e}")
                self.send_error(500, str(e))
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/api/interpretations':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                interpretations = {'interpretations': {}}
                if os.path.exists('interpretations.json'):
                    with open('interpretations.json', 'r') as f:
                        interpretations = json.load(f)
                
                feature_key = data.get('featureKey')
                interpretations['interpretations'][feature_key] = {
                    'text': data.get('text', ''),
                    'starred': data.get('starred', False),
                    'skipped': data.get('skipped', False),
                    'lastModified': datetime.now().isoformat()
                }
                
                with open('interpretations.json', 'w') as f:
                    json.dump(interpretations, f, indent=2)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'success': True}).encode())
                
                print(f"Saved interpretation for {feature_key}")
                
            except Exception as e:
                print(f"Error: {e}")
                self.send_error(500, str(e))
        else:
            self.send_error(404)

if __name__ == '__main__':
    PORT = 8085
    server = HTTPServer(('0.0.0.0', PORT), APIHandler)
    print(f"API server running on port {PORT}")
    print("Run the dashboard with: python3 -m http.server 8080")
    print("Then access: http://localhost:8080/lora_activations_dashboard.html")
    server.serve_forever()