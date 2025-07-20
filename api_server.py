#!/usr/bin/env python3
"""
Standalone API server for interpretations.
Run this alongside python -m http.server
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
from datetime import datetime

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