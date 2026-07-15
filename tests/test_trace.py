"""Tests for BoardGeometry drawing coordinates and TraceRecorder."""

import json
import random

import pytest

from catan_rl.bots import greedy_bot
from catan_rl.env.action_mask import legal_action_mask
from catan_rl.env.board import BoardConfig, BoardGeometry
from catan_rl.env.game_state import GameState
from catan_rl.env.rules import apply_action
from catan_rl.env.rules_profile import SIMPLIFIED_V1
from catan_rl.env.trace import TraceRecorder

MAX_PLIES = 5000
SEED = 5


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_geometry_has_vertex_positions_and_hex_centers():
    geo = BoardGeometry.build()
    assert len(geo.vertex_positions) == 54
    assert len(geo.hex_centers) == 19


def test_geometry_hex_center_is_mean_of_its_vertices():
    geo = BoardGeometry.build()
    for hex_id in range(geo.n_hexes):
        vids = geo.hex_to_vertices[hex_id]
        xs = [geo.vertex_positions[v][0] for v in vids]
        ys = [geo.vertex_positions[v][1] for v in vids]
        cx, cy = geo.hex_centers[hex_id]
        assert sum(xs) / 6 == pytest.approx(cx, abs=1e-6)
        assert sum(ys) / 6 == pytest.approx(cy, abs=1e-6)


# ---------------------------------------------------------------------------
# TraceRecorder: play a full seeded greedy-bot game once, reuse across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def played_game():
    rng = random.Random(SEED)
    config = BoardConfig.standard(seed=SEED)
    state = GameState.new_game(config, n_players=4, seed=SEED, profile=SIMPLIFIED_V1)

    recorder = TraceRecorder()
    recorder.start(state, {"seed": SEED, "note": "test game"})
    initial_state_dict = state.to_dict()

    plies = 0
    while not state.is_terminal and plies < MAX_PLIES:
        action = greedy_bot.pick_action(state, rng)
        apply_action(state, action, rng)
        recorder.record(action, state)
        plies += 1

    assert state.is_terminal, "seeded greedy game did not finish"
    return {
        "recorder": recorder,
        "final_state": state,
        "config": config,
        "initial_state_dict": initial_state_dict,
    }


def test_full_game_round_trip_last_ply_matches_live_state(played_game):
    recorder = played_game["recorder"]
    final_state = played_game["final_state"]
    trace = recorder.to_dict()
    assert trace["plies"], "expected at least one recorded ply"
    assert trace["plies"][-1]["state"] == final_state.to_dict()


def test_recorded_actions_were_legal_in_preceding_state(played_game):
    recorder = played_game["recorder"]
    config = played_game["config"]
    initial_state_dict = played_game["initial_state_dict"]
    plies = recorder.to_dict()["plies"]

    prev_state_dicts = [initial_state_dict] + [p["state"] for p in plies[:-1]]

    sample_indices = set(range(0, len(plies), 10))
    sample_indices.add(0)
    sample_indices.add(len(plies) - 1)

    for i in sorted(sample_indices):
        rebuilt = GameState.from_dict(prev_state_dicts[i], config)
        mask = legal_action_mask(rebuilt)
        action_index = plies[i]["action_index"]
        assert mask[action_index], (
            f"ply {i}: action_index {action_index} was not legal in preceding state"
        )


def test_trace_is_json_serializable(played_game):
    recorder = played_game["recorder"]
    json.dumps(recorder.to_dict())  # must not raise


def test_save_writes_loadable_json_and_creates_parent_dirs(played_game, tmp_path):
    recorder = played_game["recorder"]
    out_path = tmp_path / "nested" / "dir" / "trace.json"
    result_path = recorder.save(out_path)

    assert result_path == out_path
    assert out_path.exists()

    with open(out_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["version"] == 1
    geo = loaded["header"]["geometry"]
    assert len(geo["vertex_positions"]) == 54
    assert len(geo["hex_centers"]) == 19
    assert len(geo["edge_to_vertices"]) == 72
    assert len(geo["hex_to_vertices"]) == 19
    for vids in geo["hex_to_vertices"]:
        assert len(vids) == 6

    board = loaded["header"]["board"]
    assert len(board["hex_resources"]) == 19
    assert len(board["hex_tokens"]) == 19
