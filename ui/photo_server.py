#!/usr/bin/env python3
"""
Simple photo server for Remember Twelve viewer.
Serves the viewer.html and loads photos from the curation JSON file.
"""

import json
import os
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

class PhotoHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom handler that can serve photos from anywhere on disk."""

    def __init__(self, *args, **kwargs):
        # Load the curation data
        json_file = Path(__file__).parent.parent / "twelve_2023_balanced.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                self.curation_data = json.load(f)
        else:
            self.curation_data = None

        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        # Parse the path
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        # Handle photo requests
        if path.startswith('/photo/'):
            photo_index = int(path.split('/photo/')[1])
            if self.curation_data and 0 <= photo_index < len(self.curation_data['photos']):
                photo_path = Path(self.curation_data['photos'][photo_index]['photo_path'])
                if photo_path.exists():
                    self.serve_photo(photo_path)
                    return

        # Default behavior for other files
        return super().do_GET()

    def serve_photo(self, photo_path: Path):
        """Serve a photo file."""
        try:
            # Determine content type
            ext = photo_path.suffix.lower()
            content_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.heic': 'image/heic',
                '.heif': 'image/heif'
            }
            content_type = content_types.get(ext, 'application/octet-stream')

            # Read and send the file
            with open(photo_path, 'rb') as f:
                content = f.read()

            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)

        except Exception as e:
            self.send_error(500, f"Error serving photo: {e}")

def main():
    """Start the photo server."""
    port = 8080
    server_address = ('', port)
    httpd = HTTPServer(server_address, PhotoHTTPRequestHandler)

    print(f"🌐 Remember Twelve Photo Server")
    print(f"━" * 50)
    print(f"📷 Server running at: http://localhost:{port}")
    print(f"🖼️  Open viewer at: http://localhost:{port}/viewer_dynamic.html")
    print(f"⌨️  Press Ctrl+C to stop")
    print(f"━" * 50)
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down server...")
        httpd.shutdown()

if __name__ == '__main__':
    main()
