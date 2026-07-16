"""Tests for the Flask dashboard backend (catan_rl.dashboard.app)."""

import json
import random

import pytest

from catan_rl.bots import greedy_bot
from catan_rl.env.board import BoardConfig
from catan_rl.env.game_state import GameState
from catan_rl.env.rules import apply_action
from catan_rl.env.rules_profile import RulesProfile
from catan_rl.env.trace import TraceRecorder

from catan_rl.dashboard.app import create_app

SEED = 7
FAST_PROFILE = RulesProfile(name="fast", dev_cards_enabled=False, win_vp=8)
MAX_PLIES = 5000


def _play_traced_game(seed, meta):
    rng = random.Random(seed)
    config = BoardConfig.standard(seed=seed)
    state = GameState.new_game(config, n_players=4, seed=seed, profile=FAST_PROFILE)

    recorder = TraceRecorder()
    recorder.start(state, meta)

    plies = 0
    while not state.is_terminal and plies < MAX_PLIES:
        action = greedy_bot.pick_action(state, rng)
        apply_action(state, action, rng)
        recorder.record(action, state)
        plies += 1

    assert state.is_terminal, "seeded greedy game did not finish"
    return recorder


@pytest.fixture(scope="module")
def trace_dict():
    recorder = _play_traced_game(SEED, {"seed": SEED, "seats": ["a", "b", "c", "d"]})
    return recorder.to_dict()


@pytest.fixture()
def runs_dir(tmp_path, trace_dict):
    """runs_dir / run_a / traces / game0000.json (with seats meta)
    runs_dir / run_b / traces / game0000.json (no seats meta -> default names)
    runs_dir / not_a_run / (no traces dir at all)
    """
    run_a_traces = tmp_path / "run_a" / "traces"
    run_a_traces.mkdir(parents=True)
    with open(run_a_traces / "game0000.json", "w", encoding="utf-8") as f:
        json.dump(trace_dict, f)

    # second trace file in run_a, no seats in meta -> default seat names
    recorder_b = _play_traced_game(SEED + 1, {"seed": SEED + 1})
    trace_b = recorder_b.to_dict()
    with open(run_a_traces / "iter0000_game0001.json", "w", encoding="utf-8") as f:
        json.dump(trace_b, f)

    run_b_traces = tmp_path / "run_b" / "traces"
    run_b_traces.mkdir(parents=True)
    with open(run_b_traces / "game0000.json", "w", encoding="utf-8") as f:
        json.dump(trace_dict, f)

    # A directory that is not a run (no traces/ subdir) must be ignored.
    (tmp_path / "not_a_run").mkdir()

    # A secret file sibling to runs_dir, used for traversal attempts.
    with open(tmp_path / "secret.json", "w", encoding="utf-8") as f:
        json.dump({"secret": True}, f)

    return tmp_path


@pytest.fixture()
def client(runs_dir):
    app = create_app(runs_dir)
    app.config.update(TESTING=True)
    return app.test_client()


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

def test_index_serves_placeholder_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<h1" in resp.data or b"<H1" in resp.data.upper()


# ---------------------------------------------------------------------------
# GET /api/runs
# ---------------------------------------------------------------------------

def test_list_runs_returns_only_dirs_with_traces(client):
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.get_json()
    by_name = {r["run"]: r for r in data}
    assert set(by_name) == {"run_a", "run_b"}
    assert by_name["run_a"]["n_traces"] == 2
    assert by_name["run_b"]["n_traces"] == 1


def test_list_runs_on_missing_runs_dir_is_graceful(tmp_path):
    app = create_app(tmp_path / "does_not_exist")
    client = app.test_client()
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    assert resp.get_json() == []


# ---------------------------------------------------------------------------
# GET /api/traces/<run>
# ---------------------------------------------------------------------------

def test_list_traces_summaries(client, trace_dict):
    resp = client.get("/api/traces/run_a")
    assert resp.status_code == 200
    data = resp.get_json()
    by_file = {t["file"]: t for t in data}
    assert set(by_file) == {"game0000.json", "iter0000_game0001.json"}

    game0 = by_file["game0000.json"]
    expected_last_state = trace_dict["plies"][-1]["state"]
    assert game0["turns"] == expected_last_state["turn_number"]
    assert game0["winner"] == expected_last_state["winner"]
    assert game0["seats"] == ["a", "b", "c", "d"]

    game1 = by_file["iter0000_game0001.json"]
    assert game1["seats"] == ["player_0", "player_1", "player_2", "player_3"]


def test_list_traces_for_missing_run_is_404(client):
    resp = client.get("/api/traces/does_not_exist")
    assert resp.status_code == 404


def test_list_traces_skips_corrupt_trace_file(client, runs_dir):
    bad_path = runs_dir / "run_a" / "traces" / "bad.json"
    with open(bad_path, "w", encoding="utf-8") as f:
        f.write("{not json")

    resp = client.get("/api/traces/run_a")
    assert resp.status_code == 200
    data = resp.get_json()
    by_file = {t["file"]: t for t in data}
    assert "bad.json" not in by_file
    assert {"game0000.json", "iter0000_game0001.json"} <= set(by_file)


# ---------------------------------------------------------------------------
# GET /api/trace/<run>/<file>
# ---------------------------------------------------------------------------

def test_fetch_trace_matches_file_on_disk(client, runs_dir):
    resp = client.get("/api/trace/run_a/game0000.json")
    assert resp.status_code == 200
    with open(runs_dir / "run_a" / "traces" / "game0000.json", "r", encoding="utf-8") as f:
        on_disk = json.load(f)
    assert resp.get_json() == on_disk


def test_fetch_trace_missing_file_is_404(client):
    resp = client.get("/api/trace/run_a/nope.json")
    assert resp.status_code == 404


def test_fetch_trace_missing_run_is_404(client):
    resp = client.get("/api/trace/nope/game0000.json")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------

def test_traversal_in_file_position_is_rejected(client):
    resp = client.get("/api/trace/run_a/..%2f..%2fsecret.json")
    assert resp.status_code in (400, 404)


def test_traversal_in_run_position_is_rejected(client):
    resp = client.get("/api/trace/..%2f..%2f/secret.json")
    assert resp.status_code in (400, 404)


def test_safe_path_helper_rejects_dotdot_segment(runs_dir):
    # Direct unit test of the resolve()+is_relative_to guard the handlers use,
    # independent of whatever encoding/normalization werkzeug's router applies
    # before a request ever reaches the view function.
    from catan_rl.dashboard.app import _safe_path

    assert _safe_path(runs_dir, "run_a", "..", "..", "secret.json") is None
    assert _safe_path(runs_dir, "run_a") == (runs_dir / "run_a").resolve()


def test_traversal_in_traces_list_run_is_rejected(client):
    resp = client.get("/api/traces/..%2f..%2f")
    assert resp.status_code in (400, 404)


# ---------------------------------------------------------------------------
# Backslash traversal (%5c) -- these reach the view function (unlike the
# %2f tests above, which werkzeug 404s at routing before any handler code
# runs) so they actually exercise `_safe_path` / `_is_safe_segment`.
# ---------------------------------------------------------------------------

def test_backslash_traversal_in_file_position_is_rejected_and_does_not_leak(client):
    # run_a/traces/../../secret.json resolves to <runs_dir>/secret.json,
    # which is planted by the `runs_dir` fixture as a sibling of run_a/run_b.
    resp = client.get("/api/trace/run_a/..%5c..%5csecret.json")
    assert resp.status_code in (400, 404)
    assert b'"secret": true' not in resp.data.lower()
    assert resp.get_json() != {"secret": True}


def test_backslash_traversal_cannot_cross_into_another_run(client):
    # run_a/traces/../../run_b/traces/game0000.json resolves into run_b's
    # traces dir -- must not be reachable through run_a's URL.
    resp = client.get("/api/trace/run_a/..%5c..%5crun_b%5ctraces%5cgame0000.json")
    assert resp.status_code in (400, 404)


def test_backslash_traversal_in_run_position_is_rejected(client):
    resp = client.get("/api/trace/..%5c..%5csecret.json/whatever.json")
    assert resp.status_code in (400, 404)


def test_backslash_traversal_in_traces_list_run_is_rejected(client):
    resp = client.get("/api/traces/..%5c..%5c")
    assert resp.status_code in (400, 404)


def test_is_safe_segment_rejects_bad_segments():
    from catan_rl.dashboard.app import _is_safe_segment

    for bad in ("..", r"a\b", "a/b", "C:x", ""):
        assert _is_safe_segment(bad) is False, bad
    assert _is_safe_segment("game0000.json") is True
    assert _is_safe_segment("run_a") is True
