"""
Observation generator for the Catan environment.

Modes:
  "self_play" (Mode B) -- exact self info, public opponent info; OBS_DIM = 1520
  "perfect"   (Mode A) -- all players' exact info exposed;      OBS_DIM_PERFECT = 1565

The observation is always encoded relative to the observing player so that a
shared policy sees a consistent layout regardless of seat position.
Player slot 0 in the vector always corresponds to the observer; slots 1-3 are
next-clockwise opponents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .actions import DevCard, Resource
from .game_state import Phase

if TYPE_CHECKING:
    from .game_state import GameState

# Board constants
_N_HEX = 19
_N_VERTEX = 54
_N_EDGE = 72
_N_PLAYERS = 4
_N_RESOURCES = 5
_N_DEV_CARDS = 5
_N_PHASES = 12  # Phase enum has 12 members

# Segment sizes
_SEG_HEX_RESOURCES = _N_HEX * 6        # 114  one-hot terrain type per hex
_SEG_HEX_TOKENS    = _N_HEX            # 19   normalized dice token per hex
_SEG_PORT_VERTICES = _N_VERTEX * 7     # 378  one-hot port type per vertex
_SEG_ROBBER        = _N_HEX            # 19   one-hot robber hex
_SEG_ROADS         = _N_EDGE * 5       # 360  one-hot road owner (0=none)
_SEG_SETTLEMENTS   = _N_VERTEX * 5     # 270  one-hot settlement owner
_SEG_CITIES        = _N_VERTEX * 5     # 270  one-hot city owner
_SEG_PUBLIC        = _N_PLAYERS * 14   # 56   public per-player features (rotated)
_SEG_SELF_PRIV     = _N_RESOURCES * 3  # 15   self resources / dev_cards / new_cards
_SEG_CTX           = 4 + _N_PHASES + 2 + 1  # 19  turn context

OBS_DIM = (
    _SEG_HEX_RESOURCES + _SEG_HEX_TOKENS + _SEG_PORT_VERTICES
    + _SEG_ROBBER + _SEG_ROADS + _SEG_SETTLEMENTS + _SEG_CITIES
    + _SEG_PUBLIC + _SEG_SELF_PRIV + _SEG_CTX
)  # = 1520

# Mode A adds the 3 opponents' private info (15 floats each)
OBS_DIM_PERFECT = OBS_DIM + 3 * _SEG_SELF_PRIV  # = 1565

# Port type codes: 0=none, 1=generic 3:1, 2-6=resource-specific 2:1
_PT_NONE, _PT_GENERIC = 0, 1
_PT_RES_OFFSET = 2  # Resource(0..4) + 2 gives port-type index


def _port_type(config, vertex_id: int) -> int:
    port = config.port_for_vertex(vertex_id)
    if port is None:
        return _PT_NONE
    if port.resource is None:
        return _PT_GENERIC
    return int(port.resource) + _PT_RES_OFFSET


def _encode_private(player) -> np.ndarray:
    """15-element float32 vector of exact private hand info."""
    v = np.zeros(15, dtype=np.float32)
    for r in Resource:
        v[int(r)] = player.resources[int(r)] / 19.0
    for c in DevCard:
        v[5 + int(c)] = player.dev_cards[int(c)] / 14.0
    for c in DevCard:
        v[10 + int(c)] = player.dev_cards_new[int(c)] / 14.0
    return v


def make_observation(state: "GameState", observer: int, mode: str = "self_play") -> np.ndarray:
    """
    Build a fixed-size float32 observation for the given observer seat.

    observer: 0-3 player seat index
    mode: "self_play" → OBS_DIM=1520; "perfect" → OBS_DIM_PERFECT=1565
    """
    config = state.config
    geo = config.geometry

    parts: list[np.ndarray] = []

    # ---- hex resources: 19 × 6 one-hot ----
    hex_res = np.zeros(_SEG_HEX_RESOURCES, dtype=np.float32)
    for h in range(_N_HEX):
        hex_res[h * 6 + int(config.hex_resources[h])] = 1.0
    parts.append(hex_res)

    # ---- hex tokens: 19 scalars ----
    parts.append(np.array([config.hex_tokens[h] / 12.0 for h in range(_N_HEX)], dtype=np.float32))

    # ---- port types per vertex: 54 × 7 one-hot ----
    port_v = np.zeros(_SEG_PORT_VERTICES, dtype=np.float32)
    for v in range(_N_VERTEX):
        port_v[v * 7 + _port_type(config, v)] = 1.0
    parts.append(port_v)

    # ---- robber position: 19 one-hot ----
    robber = np.zeros(_N_HEX, dtype=np.float32)
    robber[state.robber_hex] = 1.0
    parts.append(robber)

    # ---- roads: 72 × 5 one-hot (slot 0 = empty) ----
    roads = np.zeros(_SEG_ROADS, dtype=np.float32)
    all_roads = state.all_road_edges()
    for e in range(_N_EDGE):
        if e in all_roads:
            # Rotate player index relative to observer
            rel = (all_roads[e] - observer) % _N_PLAYERS
            roads[e * 5 + 1 + rel] = 1.0
        else:
            roads[e * 5 + 0] = 1.0
    parts.append(roads)

    # ---- settlements: 54 × 5 one-hot ----
    sett = np.zeros(_SEG_SETTLEMENTS, dtype=np.float32)
    all_s = state.all_settlement_vertices()
    for v in range(_N_VERTEX):
        if v in all_s:
            rel = (all_s[v] - observer) % _N_PLAYERS
            sett[v * 5 + 1 + rel] = 1.0
        else:
            sett[v * 5 + 0] = 1.0
    parts.append(sett)

    # ---- cities: 54 × 5 one-hot ----
    city = np.zeros(_SEG_CITIES, dtype=np.float32)
    all_c = state.all_city_vertices()
    for v in range(_N_VERTEX):
        if v in all_c:
            rel = (all_c[v] - observer) % _N_PLAYERS
            city[v * 5 + 1 + rel] = 1.0
        else:
            city[v * 5 + 0] = 1.0
    parts.append(city)

    # ---- public player features: 4 × 14 (rotated so observer = slot 0) ----
    pub = np.zeros(_SEG_PUBLIC, dtype=np.float32)
    for rel_i in range(_N_PLAYERS):
        pid = (observer + rel_i) % _N_PLAYERS
        p = state.players[pid]
        b = rel_i * 14
        pub[b + 0]  = p.public_vp / 10.0
        pub[b + 1]  = p.total_resources / 19.0
        pub[b + 2]  = sum(p.dev_cards) / 14.0
        pub[b + 3]  = p.army_size / 14.0
        pub[b + 4]  = p.roads_built / 15.0
        pub[b + 5]  = p.settlements_built / 5.0
        pub[b + 6]  = p.cities_built / 4.0
        pub[b + 7]  = p.roads_available / 15.0
        pub[b + 8]  = p.settlements_available / 5.0
        pub[b + 9]  = p.cities_available / 4.0
        pub[b + 10] = 1.0 if state.longest_road_holder == pid else 0.0
        pub[b + 11] = 1.0 if state.largest_army_holder == pid else 0.0
        pub[b + 12] = p.played_dev_cards[int(DevCard.KNIGHT)] / 14.0
        pub[b + 13] = sum(p.played_dev_cards) / 14.0
    parts.append(pub)

    # ---- self private features: 15 ----
    parts.append(_encode_private(state.players[observer]))

    # ---- turn context: current_player(4) + phase(12) + dice(2) + is_setup(1) ----
    ctx = np.zeros(_SEG_CTX, dtype=np.float32)
    rel_cur = (state.current_player - observer) % _N_PLAYERS
    ctx[rel_cur] = 1.0
    phase_int = int(state.phase)
    if phase_int < _N_PHASES:
        ctx[4 + phase_int] = 1.0
    if state.dice is not None:
        ctx[4 + _N_PHASES + 0] = state.dice[0] / 6.0
        ctx[4 + _N_PHASES + 1] = state.dice[1] / 6.0
    ctx[4 + _N_PHASES + 2] = 1.0 if phase_int <= 3 else 0.0
    parts.append(ctx)

    obs = np.concatenate(parts)

    if mode == "perfect":
        extras = []
        for rel_i in range(1, _N_PLAYERS):
            pid = (observer + rel_i) % _N_PLAYERS
            extras.append(_encode_private(state.players[pid]))
        obs = np.concatenate([obs] + extras)

    return obs
