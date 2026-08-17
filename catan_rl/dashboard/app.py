"""
Flask backend for the trace-browsing dashboard.

Serves a small JSON API over recorded game traces (see catan_rl.env.trace)
plus the static frontend built in a later task. Routes:

  GET /                       -> static/index.html (placeholder until built)
  GET /api/runs               -> [{"run": name, "n_traces": int}, ...]
  GET /api/traces/<run>       -> [{"file", "turns", "winner", "seats"}, ...]
  GET /api/trace/<run>/<file> -> full trace JSON

Every URL segment used to build a filesystem path is first validated as a
single, clean path component (no slash, backslash, "..", drive letters,
etc. -- see `_is_safe_segment`), then the joined path is resolved and re-checked
against `runs_dir` (and, per-handler, against the specific parent
directory it's expected to live in) before being touched. Malformed or
escaping segments get a 400 rather than reading outside the run directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from flask import Flask, abort, jsonify, send_from_directory

DEFAULT_SEATS = [f"player_{i}" for i in range(4)]

STATIC_DIR = Path(__file__).parent / "static"


def _is_safe_segment(segment: str) -> bool:
    """True iff `segment` is a single, unremarkable path component.

    Rejects empty strings, null bytes, embedded separators (``/`` or
    ``\\``), ``.``/``..``, and anything with a drive/anchor (e.g. ``C:x``)
    or that otherwise doesn't collapse to exactly one path part.

    This matters on Windows because Werkzeug only splits the request URL
    on literal ``/``, so a `%5c`-encoded segment arrives at the view
    function as a single string containing a backslash. WindowsPath then
    treats that backslash as a directory separator, so a naive
    resolve()+containment check (which only guards against escaping
    `base` as a whole) can still
    let one URL segment smuggle in several path components and traverse
    between run directories. Rejecting anything but a clean single
    component closes that off before a path is ever built.
    """
    if not segment or "\x00" in segment:
        return False
    if "/" in segment or "\\" in segment:
        return False
    if segment in (".", ".."):
        return False
    p = Path(segment)
    if p.is_absolute() or p.anchor:
        return False
    if len(p.parts) != 1:
        return False
    return True


def _safe_path(base: Path, *parts: str) -> Optional[Path]:
    """Resolve base/parts and return it iff still inside resolved base.

    Every element of `parts` must be a single, clean path component (see
    `_is_safe_segment`) -- otherwise this returns None immediately. As
    defense in depth, the joined path is also resolved and re-checked
    against `base` (e.g. in case a symlink under `base` points outside
    it), so callers can turn either failure into a 400 response.
    """
    for part in parts:
        if not _is_safe_segment(part):
            return None

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
        if run_dir is None or run_dir.parent != Path(runs_dir).resolve():
            abort(400)
        traces_dir = run_dir / "traces"
        if not traces_dir.is_dir():
            abort(404)
        summaries = []
        for f in sorted(traces_dir.glob("*.json")):
            try:
                summaries.append(_trace_summary(f))
            except (json.JSONDecodeError, KeyError, TypeError, IndexError, OSError):
                continue
        return jsonify(summaries)

    @app.get("/api/trace/<run>/<file>")
    def get_trace(run, file):
        traces_dir = _safe_path(runs_dir, run, "traces")
        if traces_dir is None:
            abort(400)
        path = _safe_path(runs_dir, run, "traces", file)
        if path is None or path.parent != traces_dir:
            abort(400)
        if not path.is_file():
            abort(404)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return jsonify({"error": "trace file is corrupt or unreadable"}), 404
        return jsonify(data)

    return app
