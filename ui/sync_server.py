#!/usr/bin/env python3
"""
Lightweight sync server for Remember-Twelve viewer.
Wraps CLI tool and provides web endpoint for syncing.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).parent.parent

@app.route('/sync', methods=['POST'])
def sync_photos():
    """Trigger Google Photos sync."""
    try:
        year = request.json.get('year', 2023)

        # Execute sync script
        result = subprocess.run(
            [f'{BASE_DIR}/sync_photos.sh', str(year)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            return jsonify({
                'status': 'success',
                'message': f'Synced photos for {year}',
                'output': result.stdout
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Sync failed',
                'error': result.stderr
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({
            'status': 'error',
            'message': 'Sync timeout (>5 minutes)'
        }), 504
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/status', methods=['GET'])
def sync_status():
    """Check if photos exist."""
    photos_dir = BASE_DIR / 'ui' / 'photos'
    photo_count = len(list(photos_dir.glob('*'))) if photos_dir.exists() else 0

    return jsonify({
        'photo_count': photo_count,
        'photos_exist': photo_count > 0
    })

if __name__ == '__main__':
    print("🌐 Remember Twelve Sync Server")
    print("━" * 50)
    print("📷 Sync endpoint: http://localhost:5002/sync")
    print("📊 Status endpoint: http://localhost:5002/status")
    print("⌨️  Press Ctrl+C to stop")
    print("━" * 50)

    app.run(host='0.0.0.0', port=5002, debug=True)
