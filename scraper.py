"""
colonist.io game scraper using Playwright WebSocket interception.

Protocol (from shared.js reverse engineering):
  Binary message format (ArrayBuffer):
    Byte 0: routing type (2=RouteToServerType, 3=RouteToServerDirect, 4=SocketRouter)
    Byte 1: message ID (numeric)
    Byte 2: N = length of channel/room name string
    Bytes 3..3+N: channel/room name (UTF-8)
    Bytes 3+N..: JSON payload (UTF-8)

  WebSocket server: wss://socket.svr.colonist.io/
  Debug flag: window.socketDebugActive = true (enables console logging)

Usage:
    python scraper.py                     # watch a live game (prompts for game ID)
    python scraper.py --game 4496         # watch game 4496
    python scraper.py --replay            # record to SQLite db
"""

import asyncio
import json
import sqlite3
import argparse
import time
from pathlib import Path
from playwright.async_api import async_playwright, WebSocket, Page


DB_PATH = Path(__file__).parent / "games.db"


def init_db(db: sqlite3.Connection):
    db.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            started_at REAL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_pk INTEGER REFERENCES games(id),
            ts REAL,
            direction TEXT,  -- 'recv' or 'send'
            routing INTEGER,
            msg_id INTEGER,
            channel TEXT,
            payload TEXT
        )
    """)
    db.commit()


def decode_message(data: bytes) -> dict:
    """Decode colonist.io binary WebSocket frame."""
    if len(data) < 3:
        return {"raw": list(data)}
    routing = data[0]
    msg_id = data[1]
    chan_len = data[2]
    chan_end = 3 + chan_len
    channel = data[3:chan_end].decode("utf-8", errors="replace")
    payload_bytes = data[chan_end:]
    try:
        payload = json.loads(payload_bytes.decode("utf-8", errors="replace"))
    except Exception:
        payload = payload_bytes.hex()
    return {
        "routing": routing,
        "msg_id": msg_id,
        "channel": channel,
        "payload": payload,
    }


def print_message(direction: str, msg: dict, ts: float):
    arrow = "<<" if direction == "recv" else ">>"
    channel = msg.get("channel", "")
    msg_id = msg.get("msg_id", "?")
    payload = msg.get("payload", "")
    # Summarize game-state-relevant fields
    summary = ""
    if isinstance(payload, dict):
        keys = list(payload.keys())[:5]
        summary = ", ".join(keys)
    print(f"[{ts:.3f}] {arrow} id={msg_id:3d} chan={channel!r:20s} | {summary}")


async def watch_game(page: Page, game_id: str, db: sqlite3.Connection | None = None):
    game_pk = None
    if db:
        cur = db.execute(
            "INSERT INTO games (game_id, started_at) VALUES (?, ?)",
            (game_id, time.time()),
        )
        db.commit()
        game_pk = cur.lastrowid

    messages: list[dict] = []

    def on_ws(ws: WebSocket):
        print(f"[WebSocket connected] {ws.url}")

        def on_frame_received(payload):
            ts = time.time()
            if isinstance(payload, bytes):
                msg = decode_message(payload)
            else:
                # text frame
                msg = {"payload": payload, "msg_id": -1, "routing": -1, "channel": ""}
            print_message("recv", msg, ts)
            messages.append({"ts": ts, "direction": "recv", **msg})
            if db and game_pk:
                db.execute(
                    "INSERT INTO messages (game_pk, ts, direction, routing, msg_id, channel, payload) VALUES (?,?,?,?,?,?,?)",
                    (
                        game_pk,
                        ts,
                        "recv",
                        msg.get("routing"),
                        msg.get("msg_id"),
                        msg.get("channel", ""),
                        json.dumps(msg.get("payload")),
                    ),
                )
                db.commit()

        def on_frame_sent(payload):
            ts = time.time()
            if isinstance(payload, bytes):
                msg = decode_message(payload)
            else:
                msg = {"payload": payload, "msg_id": -1, "routing": -1, "channel": ""}
            print_message("send", msg, ts)

        ws.on("framereceived", on_frame_received)
        ws.on("framesent", on_frame_sent)
        ws.on("close", lambda: print(f"[WebSocket closed] {ws.url}"))

    page.on("websocket", on_ws)

    url = f"https://colonist.io/#game{game_id}" if game_id else "https://colonist.io"
    print(f"Navigating to {url} ...")
    await page.goto(url, wait_until="domcontentloaded")

    # Enable socket debug logging in-page
    await page.evaluate("() => { window.socketDebugActive = true; }")

    print("Watching for messages. Press Ctrl+C to stop.\n")
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass

    return messages


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="", help="Game ID to watch (e.g. 4496)")
    parser.add_argument("--save", action="store_true", help="Save messages to games.db")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    args = parser.parse_args()

    db = None
    if args.save:
        db = sqlite3.connect(DB_PATH)
        init_db(db)
        print(f"Saving to {DB_PATH}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=args.headless)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            await watch_game(page, args.game, db)
        except KeyboardInterrupt:
            pass
        finally:
            await browser.close()
            if db:
                db.close()


if __name__ == "__main__":
    asyncio.run(main())
