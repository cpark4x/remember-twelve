#!/usr/bin/env python3
"""
Remember Twelve - Main Application Entry Point

A unified web application for curating and viewing your year in 12 photos.
Combines photo management, curation, and viewing in a single interface.

Usage:
    python remember_twelve_app.py start           # Start the server
    python remember_twelve_app.py start --no-browser  # Start without opening browser
    python remember_twelve_app.py --help          # Show help
"""

import sys
import argparse
import webbrowser
from pathlib import Path


def init_database():
    from src.database import init_db

    print("Initializing database...")
    init_db()
    print("Database initialized successfully")


def start_server(open_browser: bool = True, host: str = "0.0.0.0", port: int = 8000):
    init_database()

    print(f"\n{'=' * 60}")
    print("Remember Twelve - Photo Curation Application")
    print(f"{'=' * 60}\n")
    print(f"Starting server at http://{host}:{port}")
    print(f"Viewer available at: http://localhost:{port}/")
    print(f"API docs available at: http://localhost:{port}/docs")
    print(f"\nPress CTRL+C to stop the server\n")
    print(f"{'=' * 60}\n")

    if open_browser:
        import threading
        import time

        def open_browser_delayed():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=open_browser_delayed, daemon=True).start()

    import uvicorn
    from src.api.server import app

    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        description="Remember Twelve - Photo Curation Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python remember_twelve_app.py start
  python remember_twelve_app.py start --no-browser
  python remember_twelve_app.py start --port 8080
        """
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["start"],
        default="start",
        help="Command to execute (default: start)"
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )

    args = parser.parse_args()

    if args.command == "start":
        start_server(
            open_browser=not args.no_browser,
            host=args.host,
            port=args.port
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
