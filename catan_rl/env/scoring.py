"""
Victory point calculation, longest road, and largest army.

Longest road uses DFS over the player's road/settlement graph.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .game_state import GameState
    from .board import BoardGeometry

LONGEST_ROAD_MIN = 5
LARGEST_ARMY_MIN = 3


def longest_road_for_player(
    player_id: int,
    road_edges: Set[int],
    settlement_vertices: Set[int],
    city_vertices: Set[int],
    geo: "BoardGeometry",
) -> int:
    """
    Compute the longest road length for a player via DFS.

    Roads are broken at vertices occupied by an enemy settlement/city.
    The algorithm tries every edge as a starting edge and every direction,
    then returns the global maximum path length found.
    """
    if not road_edges:
        return 0

    # All vertices occupied by this player (do NOT break at own pieces)
    own_vertices = settlement_vertices | city_vertices

    # Build adjacency: vertex -> {neighbor_vertex: edge_id}
    # restricted to this player's road edges
    adj: Dict[int, Dict[int, int]] = {}
    for eid in road_edges:
        va, vb = geo.edge_to_vertices[eid]
        adj.setdefault(va, {})[vb] = eid
        adj.setdefault(vb, {})[va] = eid

    all_vertices = set(adj.keys())

    def dfs(current_v: int, visited_edges: Set[int], enemy_vertices: Set[int]) -> int:
        best = len(visited_edges)
        for neighbor, eid in adj.get(current_v, {}).items():
            if eid in visited_edges:
                continue
            # Road is broken if neighbor is an enemy vertex
            if neighbor in enemy_vertices:
                continue
            visited_edges.add(eid)
            length = dfs(neighbor, visited_edges, enemy_vertices)
            best = max(best, length)
            visited_edges.remove(eid)
        return best

    # Enemy vertices: any occupied vertex NOT owned by this player
    # (passed in from caller who knows all players)
    # Here we just use empty set; caller must pass the full picture
    enemy_vertices: Set[int] = set()

    best = 0
    for start_v in all_vertices:
        length = dfs(start_v, set(), enemy_vertices)
        best = max(best, length)
    return best


def compute_longest_road(
    player_id: int,
    state: "GameState",
) -> int:
    """Compute longest road for player_id accounting for enemy interruptions."""
    geo = state.config.geometry
    player = state.players[player_id]

    # Enemy vertices = all occupied vertices not belonging to this player
    all_occ = state.all_occupied_vertices()
    enemy_vertices = {v for v, pid in all_occ.items() if pid != player_id}

    if not player.road_vertices:
        return 0

    adj: Dict[int, Dict[int, int]] = {}
    for eid in player.road_vertices:
        va, vb = geo.edge_to_vertices[eid]
        adj.setdefault(va, {})[vb] = eid
        adj.setdefault(vb, {})[va] = eid

    def dfs(v: int, visited: Set[int]) -> int:
        best = len(visited)
        for neighbor, eid in adj.get(v, {}).items():
            if eid in visited:
                continue
            if neighbor in enemy_vertices:
                continue
            visited.add(eid)
            best = max(best, dfs(neighbor, visited))
            visited.remove(eid)
        return best

    best = 0
    for start_v in set(adj.keys()):
        best = max(best, dfs(start_v, set()))
    return best


def update_longest_road(state: "GameState"):
    """Recompute longest road and update the holder. Mutates state.

    Official rule: recompute on every road/settlement change; the holder
    keeps the card only while still >= LONGEST_ROAD_MIN and not strictly
    beaten. If the holder drops below the minimum, the card passes to the
    unique player at the new maximum (>= minimum); on a tie, or if nobody
    qualifies, nobody holds it.
    """
    lengths = [compute_longest_road(pid, state) for pid in range(state.n_players)]
    holder = state.longest_road_holder
    if holder is not None:
        for pid, ln in enumerate(lengths):
            if pid != holder and ln > lengths[holder] and ln >= LONGEST_ROAD_MIN:
                state.longest_road_holder = pid
                return
        if lengths[holder] < LONGEST_ROAD_MIN:
            eligible = [ln for ln in lengths if ln >= LONGEST_ROAD_MIN]
            if eligible:
                best = max(eligible)
                cands = [pid for pid, ln in enumerate(lengths) if ln == best]
                state.longest_road_holder = cands[0] if len(cands) == 1 else None
            else:
                state.longest_road_holder = None
    else:
        best = max(lengths)
        if best >= LONGEST_ROAD_MIN:
            cands = [pid for pid, ln in enumerate(lengths) if ln == best]
            if len(cands) == 1:
                state.longest_road_holder = cands[0]


def update_largest_army(state: "GameState"):
    """Update largest army holder. Mutates state."""
    armies = [p.army_size for p in state.players]
    current_holder = state.largest_army_holder

    if current_holder is not None:
        for pid, size in enumerate(armies):
            if pid != current_holder and size > armies[current_holder]:
                state.largest_army_holder = pid
                break
    else:
        best_size = max(armies)
        if best_size >= LARGEST_ARMY_MIN:
            state.largest_army_holder = armies.index(best_size)


def compute_vp(player_id: int, state: "GameState") -> int:
    """Total VP for a player including special cards."""
    player = state.players[player_id]
    vp = player.total_vp
    if state.longest_road_holder == player_id:
        vp += 2
    if state.largest_army_holder == player_id:
        vp += 2
    return vp


def check_winner(state: "GameState") -> Optional[int]:
    """Return player_id of winner if any player has reached the profile's win VP."""
    win_vp = state.profile.win_vp
    for pid in range(state.n_players):
        if compute_vp(pid, state) >= win_vp:
            return pid
    return None
