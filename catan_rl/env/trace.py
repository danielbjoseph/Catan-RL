"""
Game trace recording.

TraceRecorder captures a full game as one JSON document: a header carrying
board configuration and drawing geometry (so a replay dashboard never needs
to re-derive board topology) plus one entry per ply describing the action
taken and the resulting state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .actions import Action
from .game_state import GameState

TRACE_VERSION = 1


class TraceRecorder:
    """Records a game trace: call start() once, record() after every action."""

    def __init__(self) -> None:
        self._header: Optional[Dict[str, Any]] = None
        self._plies: List[Dict[str, Any]] = []
        self._next_actor: Optional[int] = None
        self._next_phase: Optional[str] = None

    def start(self, state: GameState, meta: dict) -> None:
        config = state.config
        geo = config.geometry
        self._header = {
            "meta": meta,
            "profile": state.profile.to_dict(),
            "board": {
                "hex_resources": [int(h) for h in config.hex_resources],
                "hex_tokens": [int(t) for t in config.hex_tokens],
                "desert_hex": int(config.desert_hex),
                "ports": [
                    {
                        "vertices": [int(port.vertices[0]), int(port.vertices[1])],
                        "resource": (
                            int(port.resource) if port.resource is not None else None
                        ),
                    }
                    for port in config.ports
                ],
                "robber_start": int(state.robber_hex),
            },
            "geometry": {
                "vertex_positions": [list(p) for p in geo.vertex_positions],
                "hex_centers": [list(c) for c in geo.hex_centers],
                "edge_to_vertices": [
                    list(geo.edge_to_vertices[e]) for e in range(geo.n_edges)
                ],
                "hex_to_vertices": [
                    list(geo.hex_to_vertices[h]) for h in range(geo.n_hexes)
                ],
                "vertex_to_hexes": [
                    list(geo.vertex_to_hexes[v]) for v in range(geo.n_vertices)
                ],
            },
        }
        self._plies = []
        self._next_actor = state.current_player
        self._next_phase = state.phase.name

    def record(self, action: Action, state: GameState) -> None:
        self._plies.append({
            "ply": len(self._plies),
            "turn": state.turn_number,
            "player": self._next_actor,
            "phase": self._next_phase,
            "action_index": int(action.catalog_index),
            "action_str": str(action),
            "dice": list(state.dice) if state.dice else None,
            "state": state.to_dict(),
        })
        self._next_actor = state.current_player
        self._next_phase = state.phase.name

    def to_dict(self) -> dict:
        return {
            "version": TRACE_VERSION,
            "header": self._header,
            "plies": self._plies,
        }

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
        return path
