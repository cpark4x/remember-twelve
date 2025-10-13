#!/usr/bin/env python3
"""
Remember Twelve - Web Viewer

A beautiful Material Design web interface to view your curated Twelve.

Usage:
    python ui/twelve_viewer.py twelve_2025_balanced.json
    python ui/twelve_viewer.py --port 8080
"""

import sys
import json
import argparse
from pathlib import Path
from flask import Flask, render_template, send_from_directory, jsonify
import webbrowser
from threading import Timer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from twelve_curator import TwelveSelection


app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# Global state
current_selection = None
photos_dir = None


@app.route('/')
def index():
    """Main gallery view."""
    if not current_selection:
        return render_template('error.html',
                             message="No Twelve selection loaded. Please provide a JSON file.")

    return render_template('gallery.html',
                         selection=current_selection,
                         year=current_selection.year)


@app.route('/api/selection')
def get_selection():
    """Get the current selection as JSON."""
    if not current_selection:
        return jsonify({'error': 'No selection loaded'}), 404

    return jsonify(current_selection.to_dict())


@app.route('/photos/<path:filename>')
def serve_photo(filename):
    """Serve photo files."""
    if not photos_dir:
        return "Photos directory not configured", 404

    return send_from_directory(photos_dir, filename)


def open_browser(port):
    """Open browser after a short delay."""
    webbrowser.open(f'http://localhost:{port}')


def main():
    parser = argparse.ArgumentParser(
        description="Remember Twelve - Web Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'selection_file',
        nargs='?',
        help='Path to Twelve selection JSON file'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to run server on (default: 5000)'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )

    args = parser.parse_args()

    global current_selection, photos_dir

    # Load selection if provided
    if args.selection_file:
        selection_path = Path(args.selection_file)

        if not selection_path.exists():
            print(f"❌ Error: File not found: {selection_path}")
            sys.exit(1)

        try:
            current_selection = TwelveSelection.load(selection_path)

            # Determine photos directory (same as first photo's parent)
            if current_selection.photos:
                first_photo = Path(current_selection.photos[0].photo_path)
                photos_dir = first_photo.parent

            print(f"✅ Loaded: {selection_path.name}")
            print(f"   Year: {current_selection.year}")
            print(f"   Photos: {len(current_selection.photos)}")
            print(f"   Strategy: {current_selection.strategy}")
            print()
        except Exception as e:
            print(f"❌ Error loading selection: {e}")
            sys.exit(1)

    # Start server
    print(f"🌐 Starting Remember Twelve Web Viewer...")
    print(f"   URL: http://localhost:{args.port}")
    print(f"   Press Ctrl+C to stop")
    print()

    # Open browser after short delay
    if not args.no_browser:
        Timer(1.5, open_browser, args=[args.port]).start()

    app.run(debug=True, port=args.port, use_reloader=False)


if __name__ == '__main__':
    main()
