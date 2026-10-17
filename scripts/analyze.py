#!/usr/bin/env python
"""
Generate the HQCT Analysis Dashboard (self-contained HTML).

Usage:
  # Paper results only
  python scripts/analyze.py

  # Include a training run
  python scripts/analyze.py --run_dir runs/ag_news_hqct-k_q6

  # Custom output path
  python scripts/analyze.py --output my_dashboard.html

  # Open in browser after generation (uses stdlib http.server)
  python scripts/analyze.py --serve
"""

import argparse
import sys
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hqct.analysis import generate_dashboard


def parse_args():
    p = argparse.ArgumentParser(description="HQCT Analysis Dashboard")
    p.add_argument("--run_dir", type=str, default=None,
                   help="Path to a training run folder containing metrics.json")
    p.add_argument("--output", type=str, default="analysis_dashboard.html",
                   help="Output HTML file path")
    p.add_argument("--no-paper", action="store_true",
                   help="Skip paper reference results")
    p.add_argument("--serve", action="store_true",
                   help="Start a local HTTP server and open the dashboard")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)

    print("Generating HQCT Analysis Dashboard …")
    path = generate_dashboard(
        run_dir=Path(args.run_dir) if args.run_dir else None,
        output_path=out,
        include_paper_results=not args.no_paper,
    )
    print(f"[OK] Dashboard written to: {path.resolve()}")

    if args.serve:
        # Serve the directory that contains the HTML
        serve_dir = path.parent.resolve()
        html_name = path.name

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=str(serve_dir), **kw)

            def log_message(self, fmt, *args):
                pass  # quiet

        server = HTTPServer(("127.0.0.1", args.port), Handler)
        url = f"http://127.0.0.1:{args.port}/{html_name}"
        print(f"Serving at {url}")
        print("Press Ctrl+C to stop.")

        # Open browser after a short delay
        def _open():
            time.sleep(0.6)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
            server.shutdown()


if __name__ == "__main__":
    main()
