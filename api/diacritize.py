"""
Vercel Serverless Function for Harakat Diacritization API
Deploy with: vercel --prod
"""

from http.server import BaseHTTPRequestHandler
import json
import sys
import os

# Add parent directory to path to import harakat
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy load harakat to reduce cold start time
_diacritize = None

def get_diacritize():
    global _diacritize
    if _diacritize is None:
        from harakat import diacritize
        _diacritize = diacritize
    return _diacritize


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Max-Age', '86400')
        self.end_headers()

    def do_POST(self):
        """Handle diacritization request"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)

            text = data.get('text', '')

            if not text:
                self._send_error(400, 'No text provided')
                return

            if len(text) > 10000:
                self._send_error(400, 'Text too long (max 10000 characters)')
                return

            # Run diacritization
            diacritize = get_diacritize()
            result = diacritize(text)

            # Send response
            self._send_json({
                'success': True,
                'input': text,
                'output': result
            })

        except json.JSONDecodeError:
            self._send_error(400, 'Invalid JSON')
        except Exception as e:
            self._send_error(500, str(e))

    def do_GET(self):
        """Health check"""
        self._send_json({
            'status': 'ok',
            'service': 'Harakat Diacritization API',
            'version': '3.5'
        })

    def _send_json(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def _send_error(self, code, message):
        """Send error response"""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({
            'success': False,
            'error': message
        }).encode('utf-8'))
