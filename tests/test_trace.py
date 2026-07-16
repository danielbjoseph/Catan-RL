"""Tests for BoardGeometry drawing coordinates and TraceRecorder."""

import json
import random

import pytest
import torch

from catan_rl.bots import greedy_bot
from catan_rl.env.action_mask import legal_action_mask
from catan_rl.env.board import BoardConfig, BoardGeometry
from catan_rl.env.game_state import GameState, Phase
from catan_rl.env.rules import apply_action
from catan_rl.env.rules_profile import SIMPLIFIED_V1, RulesProfile
from catan_rl.env.trace import TraceRecorder
from catan_rl.rl.models import ActorCritic
from catan_rl.rl.rollout import collect_rollouts

MAX_PLIES = 5000
SEED = 5
FAST_PROFILE = RulesProfile(name="fast", dev_cards_enabled=False, win_vp=8)


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

    ground_truth = []  # (current_player, phase.name) captured before each apply_action
    first_ply_bank_snapshot = None

    plies = 0
    while not state.is_terminal and plies < MAX_PLIES:
        ground_truth.append((state.current_player, state.phase.name))
        action = greedy_bot.pick_action(state, rng)
        apply_action(state, action, rng)
        recorder.record(action, state)
        if first_ply_bank_snapshot is None:
            first_ply_bank_snapshot = list(state.bank)
        plies += 1

    assert state.is_terminal, "seeded greedy game did not finish"
    return {
        "recorder": recorder,
        "final_state": state,
        "config": config,
        "initial_state_dict": initial_state_dict,
        "ground_truth": ground_truth,
        "first_ply_bank_snapshot": first_ply_bank_snapshot,
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

    for i in range(len(plies)):
        rebuilt = GameState.from_dict(prev_state_dicts[i], config)
        mask = legal_action_mask(rebuilt)
        action_index = plies[i]["action_index"]
        assert mask[action_index], (
            f"ply {i}: action_index {action_index} was not legal in preceding state"
        )


def test_recorded_bank_is_not_aliased_across_plies(played_game):
    """Regression test: GameState.to_dict() must copy self.bank.

    If to_dict() returns the live bank list by reference, every historical
    ply's recorded "state" ends up sharing the same list object, so once the
    game mutates the bank further, all previously recorded plies retroactively
    "change" to reflect the final bank instead of the bank at that point in
    time. This asserts ply 0's recorded bank still matches a snapshot taken
    immediately after it was recorded.
    """
    recorder = played_game["recorder"]
    first_ply_bank_snapshot = played_game["first_ply_bank_snapshot"]
    plies = recorder.to_dict()["plies"]

    assert plies[0]["state"]["bank"] == first_ply_bank_snapshot


def test_recorded_player_and_phase_match_ground_truth(played_game):
    recorder = played_game["recorder"]
    ground_truth = played_game["ground_truth"]
    plies = recorder.to_dict()["plies"]

    assert len(plies) == len(ground_truth)

    phase_names_seen = set()
    for i, (expected_player, expected_phase) in enumerate(ground_truth):
        assert plies[i]["player"] == expected_player, (
            f"ply {i}: recorded player {plies[i]['player']} != ground truth {expected_player}"
        )
        assert plies[i]["phase"] == expected_phase, (
            f"ply {i}: recorded phase {plies[i]['phase']} != ground truth {expected_phase}"
        )
        phase_names_seen.add(expected_phase)

    # Confirm the seeded game actually exercises discard/robber sub-phases,
    # so this test is known to cover player/phase attribution during those
    # transitions and not just the common MAIN-phase path.
    assert phase_names_seen & {Phase.DISCARD.name, Phase.ROBBER.name}, (
        "seeded game never exercised DISCARD or ROBBER phase; "
        "test would not catch attribution bugs in those phases"
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


# ---------------------------------------------------------------------------
# collect_rollouts trace_dir / trace_every wiring
# ---------------------------------------------------------------------------

def _tiny_policy():
    torch.manual_seed(0)
    return ActorCritic(hidden_sizes=(8, 8))


def test_collect_rollouts_writes_traces_when_enabled(tmp_path):
    policy = _tiny_policy()
    collect_rollouts(
        policy, n_games=2, rules_profile=FAST_PROFILE, seed=123, max_turns=500,
        trace_dir=tmp_path, trace_every=1,
    )

    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 2
    assert [f.name for f in files] == ["game0000.json", "game0001.json"]

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            trace = json.load(fh)
        assert trace["version"] == 1
        assert trace["plies"], "expected at least one recorded ply"
        last_state = trace["plies"][-1]["state"]
        # Game ended either by reaching a winner or hitting the turn cap
        # (truncation) -- either way the final recorded ply must be the
        # end-of-game snapshot.
        assert last_state["winner"] is not None or last_state["turn_number"] >= 500


def test_collect_rollouts_writes_nothing_when_trace_every_is_none(tmp_path):
    policy = _tiny_policy()
    collect_rollouts(
        policy, n_games=2, rules_profile=FAST_PROFILE, seed=123, max_turns=500,
        trace_dir=tmp_path,
    )
    assert list(tmp_path.glob("*.json")) == []


def test_collect_rollouts_respects_trace_every_stride(tmp_path):
    policy = _tiny_policy()
    collect_rollouts(
        policy, n_games=4, rules_profile=FAST_PROFILE, seed=123, max_turns=500,
        trace_dir=tmp_path, trace_every=2,
    )
    files = sorted(f.name for f in tmp_path.glob("*.json"))
    assert files == ["game0000.json", "game0002.json"]
