"""
Immutable board geometry and configuration.

BoardGeometry: computed once from hex coordinates; stores all adjacency maps.
BoardConfig: hex resources, number tokens, port locations (fixed or random).

Board uses a pointy-top hexagonal grid in axial coordinates (q, r).
Vertex/edge IDs are assigned by sorting deduplicated floating-point positions
top-to-bottom then left-to-right, giving stable deterministic ordering.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from enum import IntEnum

from .actions import Resource


class HexType(IntEnum):
    """Hex terrain type. DESERT is not a tradeable resource (use Resource for those)."""
    WOOD   = 0
    BRICK  = 1
    SHEEP  = 2
    WHEAT  = 3
    ORE    = 4
    DESERT = 5

    def to_resource(self) -> Resource:
        assert self != HexType.DESERT
        return Resource(int(self))


# ---------------------------------------------------------------------------
# Hex coordinate constants
# ---------------------------------------------------------------------------

# 19 hex positions in axial (q, r) -- radius-2 hex grid
_HEX_COORDS: List[Tuple[int, int]] = [
    # ring 0
    (0, 0),
    # ring 1
    (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
    # ring 2
    (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1),
    (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -2), (2, -1),
]

# Axial neighbor directions E, NE, NW, W, SW, SE (pointy-top)
_DIRECTIONS: List[Tuple[int, int]] = [
    (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
]

# Standard hex type distribution (19 hexes)
_STANDARD_RESOURCES: List[HexType] = (
    [HexType.DESERT]
    + [HexType.WOOD] * 4
    + [HexType.BRICK] * 3
    + [HexType.SHEEP] * 4
    + [HexType.WHEAT] * 4
    + [HexType.ORE] * 3
)

# Standard number tokens (placed on 18 non-desert hexes, sorted for shuffling)
_STANDARD_TOKENS: List[int] = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]

# Port definitions as (q, r, vi, vj, resource_or_None)
# vi, vj are vertex direction indices (0=N, 1=NE, 2=SE, 3=S, 4=SW, 5=NW) within the hex.
# resource=None means 3:1 generic port.
_PORT_DEFS: List[Tuple[int, int, int, int, Optional[Resource]]] = [
    (1,  -2, 0, 1, None),           # top-right, 3:1
    (2,  -1, 1, 2, Resource.ORE),   # right, ore 2:1
    (2,   0, 2, 3, None),           # right-lower, 3:1
    (0,   2, 3, 4, Resource.WHEAT), # bottom-right, wheat 2:1
    (-2,  2, 3, 4, None),           # bottom-left, 3:1
    (-2,  0, 4, 5, Resource.BRICK), # left, brick 2:1
    (-1, -1, 5, 0, Resource.WOOD),  # top-left, wood 2:1
    (0,  -2, 5, 0, None),           # top, 3:1
    (1,   1, 2, 3, Resource.SHEEP), # right-bottom, sheep 2:1
]


# ---------------------------------------------------------------------------
# Geometry construction helpers
# ---------------------------------------------------------------------------

def _hex_center(q: int, r: int, size: float = 1.0) -> Tuple[float, float]:
    x = size * math.sqrt(3) * (q + r / 2.0)
    y = size * 1.5 * r
    return x, y


def _hex_vertex_positions(q: int, r: int, size: float = 1.0) -> List[Tuple[float, float]]:
    """Return 6 vertex positions for hex (q,r) in order N, NE, SE, S, SW, NW."""
    cx, cy = _hex_center(q, r, size)
    positions = []
    for i in range(6):
        angle_rad = math.pi / 180.0 * (90.0 - 60.0 * i)
        vx = cx + size * math.cos(angle_rad)
        vy = cy + size * math.sin(angle_rad)
        positions.append((round(vx, 6), round(vy, 6)))
    return positions


def _round_pos(pos: Tuple[float, float]) -> Tuple[float, float]:
    return (round(pos[0], 5), round(pos[1], 5))


# ---------------------------------------------------------------------------
# BoardGeometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoardGeometry:
    """
    Precomputed adjacency maps for the Catan board.
    All IDs are stable integers derived from sorted deduplicated positions.
    """
    hex_coords: Tuple[Tuple[int, int], ...]      # (q, r) for hex i
    n_hexes: int                                   # 19
    n_vertices: int                                # 54
    n_edges: int                                   # 72

    hex_to_vertices: Dict[int, List[int]]          # hex -> [v0..v5] ordered N..NW
    hex_to_edges:    Dict[int, List[int]]          # hex -> [e0..e5]
    vertex_to_hexes: Dict[int, List[int]]          # vertex -> [hex_ids]
    vertex_to_vertices: Dict[int, List[int]]       # vertex -> neighboring vertex_ids
    vertex_to_edges: Dict[int, List[int]]          # vertex -> edge_ids
    edge_to_vertices: Dict[int, Tuple[int, int]]   # edge -> (v_a, v_b)
    edge_to_hexes:   Dict[int, List[int]]          # edge -> [hex_ids]
    coord_to_hex:    Dict[Tuple[int, int], int]    # (q,r) -> hex_id

    vertex_positions: Tuple[Tuple[float, float], ...]  # (x, y) for vertex i
    hex_centers: Tuple[Tuple[float, float], ...]       # (x, y) for hex i

    @classmethod
    def build(cls) -> BoardGeometry:
        coords = _HEX_COORDS
        coord_set = set(coords)
        coord_to_hex = {c: i for i, c in enumerate(coords)}

        # Collect all vertex positions keyed by rounded (x, y)
        pos_to_vid: dict = {}
        hex_raw_vertices: List[List[Tuple[float, float]]] = []
        for q, r in coords:
            verts = _hex_vertex_positions(q, r)
            hex_raw_vertices.append(verts)
            for p in verts:
                rp = _round_pos(p)
                if rp not in pos_to_vid:
                    pos_to_vid[rp] = None  # placeholder

        # Sort positions top-to-bottom, left-to-right and assign IDs
        sorted_positions = sorted(pos_to_vid.keys(), key=lambda p: (-p[1], p[0]))
        for vid, pos in enumerate(sorted_positions):
            pos_to_vid[pos] = vid
        n_vertices = len(sorted_positions)

        # Build hex_to_vertices, tracking a higher-precision raw position per
        # vertex id (the dedup keys above are rounded to 5 decimals, which is
        # too coarse for drawing geometry consumers that need hex-center
        # precision; the 6-decimal raw positions agree across sharing hexes).
        hex_to_vertices: Dict[int, List[int]] = {}
        raw_pos_by_vid: Dict[int, Tuple[float, float]] = {}
        for hi, (q, r) in enumerate(coords):
            verts = hex_raw_vertices[hi]
            vids = [pos_to_vid[_round_pos(p)] for p in verts]
            hex_to_vertices[hi] = vids
            for vid, p in zip(vids, verts):
                raw_pos_by_vid.setdefault(vid, p)

        # Build edges: each edge is a frozenset of two adjacent vertices on same hex
        edge_set: Dict[FrozenSet[int], int] = {}
        hex_to_edges: Dict[int, List[int]] = {i: [] for i in range(len(coords))}
        edge_to_vertices: Dict[int, Tuple[int, int]] = {}
        edge_to_hexes: Dict[int, List[int]] = {}

        for hi in range(len(coords)):
            vids = hex_to_vertices[hi]
            for i in range(6):
                va, vb = vids[i], vids[(i + 1) % 6]
                key = frozenset({va, vb})
                if key not in edge_set:
                    eid = len(edge_set)
                    edge_set[key] = eid
                    edge_to_vertices[eid] = (min(va, vb), max(va, vb))
                    edge_to_hexes[eid] = []
                eid = edge_set[key]
                hex_to_edges[hi].append(eid)
                if hi not in edge_to_hexes[eid]:
                    edge_to_hexes[eid].append(hi)
        n_edges = len(edge_set)

        # Build vertex_to_hexes, vertex_to_vertices, vertex_to_edges
        vertex_to_hexes: Dict[int, List[int]] = {v: [] for v in range(n_vertices)}
        vertex_to_vertices: Dict[int, List[int]] = {v: [] for v in range(n_vertices)}
        vertex_to_edges: Dict[int, List[int]] = {v: [] for v in range(n_vertices)}

        for hi, vids in hex_to_vertices.items():
            for v in vids:
                if hi not in vertex_to_hexes[v]:
                    vertex_to_hexes[v].append(hi)

        for eid, (va, vb) in edge_to_vertices.items():
            vertex_to_edges[va].append(eid)
            vertex_to_edges[vb].append(eid)
            if vb not in vertex_to_vertices[va]:
                vertex_to_vertices[va].append(vb)
            if va not in vertex_to_vertices[vb]:
                vertex_to_vertices[vb].append(va)

        return cls(
            hex_coords=tuple(coords),
            n_hexes=len(coords),
            n_vertices=n_vertices,
            n_edges=n_edges,
            hex_to_vertices=hex_to_vertices,
            hex_to_edges=hex_to_edges,
            vertex_to_hexes=vertex_to_hexes,
            vertex_to_vertices=vertex_to_vertices,
            vertex_to_edges=vertex_to_edges,
            edge_to_vertices=edge_to_vertices,
            edge_to_hexes=edge_to_hexes,
            coord_to_hex=coord_to_hex,
            vertex_positions=tuple(raw_pos_by_vid[vid] for vid in range(n_vertices)),
            hex_centers=tuple(_hex_center(q, r) for q, r in coords),
        )

    def hex_neighbors(self, hex_id: int) -> List[int]:
        q, r = self.hex_coords[hex_id]
        result = []
        for dq, dr in _DIRECTIONS:
            neighbor = (q + dq, r + dr)
            if neighbor in self.coord_to_hex:
                result.append(self.coord_to_hex[neighbor])
        return result

    def vertices_on_hex(self, hex_id: int) -> List[int]:
        return self.hex_to_vertices[hex_id]

    def is_coastal_vertex(self, vertex_id: int) -> bool:
        return len(self.vertex_to_hexes[vertex_id]) < 3

    def is_coastal_edge(self, edge_id: int) -> bool:
        return len(self.edge_to_hexes[edge_id]) < 2


# ---------------------------------------------------------------------------
# Port
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Port:
    vertices: Tuple[int, int]        # two vertex IDs that receive this port
    resource: Optional[Resource]     # None = generic 3:1 port
    rate: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'rate', 2 if self.resource is not None else 3)


# ---------------------------------------------------------------------------
# BoardConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoardConfig:
    """
    Complete board configuration: geometry + per-hex resources/tokens + ports.
    Immutable once created.
    """
    geometry: BoardGeometry
    hex_resources: Tuple[HexType, ...]    # len=19, indexed by hex_id
    hex_tokens: Tuple[int, ...]           # len=19, 0 for desert
    ports: Tuple[Port, ...]               # 9 ports
    desert_hex: int                        # hex_id of the desert

    # ---------------------------------------------------------------------------
    @classmethod
    def standard(cls, seed: Optional[int] = None) -> BoardConfig:
        """Fixed resource/token layout (standard Catan beginner board)."""
        rng = random.Random(seed)
        geo = BoardGeometry.build()

        resources = list(_STANDARD_RESOURCES)
        rng.shuffle(resources)

        tokens: List[int] = [0] * 19
        token_list = list(_STANDARD_TOKENS)
        rng.shuffle(token_list)
        ti = 0
        desert_hex = -1
        for hi, res in enumerate(resources):
            if res == HexType.DESERT:
                desert_hex = hi
            else:
                tokens[hi] = token_list[ti]
                ti += 1

        ports = cls._build_ports(geo)
        return cls(
            geometry=geo,
            hex_resources=tuple(resources),
            hex_tokens=tuple(tokens),
            ports=tuple(ports),
            desert_hex=desert_hex,
        )

    @classmethod
    def _build_ports(cls, geo: BoardGeometry) -> List[Port]:
        ports: List[Port] = []
        coord_to_hex = geo.coord_to_hex
        for q, r, vi, vj, resource in _PORT_DEFS:
            if (q, r) not in coord_to_hex:
                continue
            hi = coord_to_hex[(q, r)]
            vids = geo.hex_to_vertices[hi]
            va, vb = vids[vi % 6], vids[vj % 6]
            ports.append(Port(vertices=(va, vb), resource=resource))
        return ports

    def port_for_vertex(self, vertex_id: int) -> Optional[Port]:
        for port in self.ports:
            if vertex_id in port.vertices:
                return port
        return None

    def best_trade_rate(self, vertex_ids: List[int], resource: Resource) -> int:
        """Return the best maritime trade rate for a given resource given owned vertices."""
        rate = 4
        for vid in vertex_ids:
            port = self.port_for_vertex(vid)
            if port is None:
                continue
            if port.resource is None:
                rate = min(rate, 3)
            elif port.resource == resource:
                rate = min(rate, 2)
        return rate


# ---------------------------------------------------------------------------
# Module-level singleton (computed once)
# ---------------------------------------------------------------------------

_GEOMETRY: Optional[BoardGeometry] = None


def get_geometry() -> BoardGeometry:
    global _GEOMETRY
    if _GEOMETRY is None:
        _GEOMETRY = BoardGeometry.build()
    return _GEOMETRY
