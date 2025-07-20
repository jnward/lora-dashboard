# RunPod Dashboard Setup

## Using with VSCode Port Forwarding

VSCode automatically forwards ports when you're connected via Remote-SSH. Here's how to use the dashboard:

1. **Start the server in your RunPod terminal:**
   ```bash
   python3 dashboard_server.py
   ```

2. **VSCode will automatically detect and forward port 8080**
   - Look for a notification popup in VSCode
   - Or check the "Ports" tab in the VSCode terminal panel

3. **Access the dashboard:**
   - Click the forwarded port URL in VSCode
   - Or manually open: http://localhost:8080

## Alternative: SSH Port Forwarding

If not using VSCode, you can manually forward the port:

```bash
# From your local machine:
ssh -L 8080:localhost:8080 <your-runpod-ssh-connection>
```

Then access http://localhost:8080 in your browser.

## Troubleshooting

1. **Test the API is working:**
   ```bash
   python3 test_api.py
   ```

2. **Check server logs:**
   The server prints all requests, so you should see:
   - GET requests when loading the page
   - POST requests when saving interpretations

3. **Check interpretations are saved:**
   ```bash
   cat interpretations.json
   ```

## Note on File Access

The dashboard needs to be regenerated after creating new activation data:
```bash
python3 generate_html_dashboard.py
```

This ensures the interpretation UI is added to all panels.