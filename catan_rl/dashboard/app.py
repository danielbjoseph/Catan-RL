"""
Flask backend for the trace-browsing dashboard.

Serves a small JSON API over recorded game traces (see catan_rl.env.trace)
plus the static frontend built in a later task. Routes:

  GET /                       -> static/index.html (placeholder until built)
  GET /api/runs               -> [{"run": name, "n_traces": int}, ...]
  GET /api/traces/<run>       -> [{"file", "turns", "winner", "seats"}, ...]
  GET /api/trace/<run>/<file> -> full trace JSON

All filesystem paths built from URL segments are resolved and checked against
`runs_dir` with `Path.is_relative_to` before being touched, so `..` segments
(and other escapes) get a 400 rather than reading outside the run directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from flask import Flask, abort, jsonify, send_from_directory

DEFAULT_SEATS = [f"player_{i}" for i in range(4)]

STATIC_DIR = Path(__file__).parent / "static"


def _safe_path(base: Path, *parts: str) -> Optional[Path]:
    """Resolve base/parts and return it iff still inside resolved base.

    Returns None if the resulting path escapes `base` (e.g. via a ".."
    segment), so callers can turn that into a 400 response.
    """
    base = Path(base).resolve()
    candidate = base
    for part in parts:
        candidate = candidate / part
    candidate = candidate.resolve()
    if candidate != base and base not in candidate.parents:
        return None
    return candidate


def _trace_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        trace = json.load(fh)

    plies = trace.get("plies") or []
    last_state = plies[-1]["state"] if plies else {}

    header = trace.get("header") or {}
    meta = header.get("meta") or {}
    seats = meta.get("seats") or DEFAULT_SEATS

    return {
        "file": path.name,
        "turns": last_state.get("turn_number"),
        "winner": last_state.get("winner"),
        "seats": seats,
    }


def create_app(runs_dir) -> Flask:
    runs_dir = Path(runs_dir)

    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )

    @app.get("/")
    def index():
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            abort(404)
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/runs")
    def list_runs():
        runs = []
        if runs_dir.is_dir():
            for entry in sorted(runs_dir.iterdir()):
                if not entry.is_dir():
                    continue
                traces_dir = entry / "traces"
                if not traces_dir.is_dir():
                    continue
                n_traces = len(list(traces_dir.glob("*.json")))
                runs.append({"run": entry.name, "n_traces": n_traces})
        return jsonify(runs)

    @app.get("/api/traces/<run>")
    def list_traces(run):
        run_dir = _safe_path(runs_dir, run)
        if run_dir is None:
            abort(400)
        traces_dir = run_dir / "traces"
        if not traces_dir.is_dir():
            abort(404)
        summaries = [_trace_summary(f) for f in sorted(traces_dir.glob("*.json"))]
        return jsonify(summaries)

    @app.get("/api/trace/<run>/<file>")
    def get_trace(run, file):
        path = _safe_path(runs_dir, run, "traces", file)
        if path is None:
            abort(400)
        if not path.is_file():
            abort(404)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return jsonify(data)

    return app
