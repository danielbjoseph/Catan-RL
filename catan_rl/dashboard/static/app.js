"use strict";

/* ---------------------------------------------------------------------- */
/* Constants                                                              */
/* ---------------------------------------------------------------------- */

const SVGNS = "http://www.w3.org/2000/svg";

const RESOURCE_NAMES = ["wood", "brick", "sheep", "wheat", "ore"];
const RESOURCE_ABBR = ["W", "B", "S", "H", "O"];
const RES_VAR = ["--wood", "--brick", "--sheep", "--wheat", "--ore"];
const HEX_VAR = ["--wood", "--brick", "--sheep", "--wheat", "--ore", "--desert"];
const DEV_ABBR = ["Knight", "RoadBuild", "YearPlenty", "Monopoly", "VP"];
const SEAT_VAR = ["--seat-0", "--seat-1", "--seat-2", "--seat-3"];
const PLY_MS = 200; // ~5 plies/sec

/* ---------------------------------------------------------------------- */
/* App state                                                              */
/* ---------------------------------------------------------------------- */

let currentRun = null;
let currentTraceFile = null;
let trace = null;
let header = null;
let geo = null;
let board = null;
let plies = [];
let seatNames = [];
let rollEvents = [];
let currentPly = 0;
let playing = false;
let playTimer = null;

/* ---------------------------------------------------------------------- */
/* Small helpers                                                         */
/* ---------------------------------------------------------------------- */

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function svgEl(tag, attrs) {
  const el = document.createElementNS(SVGNS, tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

function seatLabel(pid) {
  if (pid === null || pid === undefined) return "—";
  return (seatNames && seatNames[pid]) || `Player ${pid}`;
}

function showView(id) {
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.id !== id; });
  document.getElementById("back-btn").hidden = id === "view-runs";
  const crumb = document.getElementById("crumb");
  if (id === "view-runs") crumb.textContent = "Catan RL Dashboard";
  else if (id === "view-traces") crumb.textContent = `Run: ${currentRun}`;
  else crumb.textContent = `${currentRun} / ${currentTraceFile}`;
}

/* ---------------------------------------------------------------------- */
/* Run browser                                                            */
/* ---------------------------------------------------------------------- */

async function loadRuns() {
  showView("view-runs");
  const tbody = document.querySelector("#runs-table tbody");
  tbody.innerHTML = "";
  const runs = await getJSON("/api/runs");
  document.getElementById("runs-empty").hidden = runs.length > 0;
  runs.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${escapeHtml(r.run)}</td><td>${r.n_traces}</td>`;
    tr.addEventListener("click", () => {
      currentRun = r.run;
      loadTraces(r.run);
    });
    tbody.appendChild(tr);
  });
}

async function loadTraces(run) {
  showView("view-traces");
  document.getElementById("traces-hint").textContent = `Loading ${run}…`;
  const tbody = document.querySelector("#traces-table tbody");
  tbody.innerHTML = "";
  const list = await getJSON(`/api/traces/${encodeURIComponent(run)}`);
  document.getElementById("traces-hint").textContent = `Run: ${run} — ${list.length} trace(s)`;
  document.getElementById("traces-empty").hidden = list.length > 0;
  list.forEach((t) => {
    const winnerLabel = t.winner === null || t.winner === undefined
      ? "—"
      : (t.seats && t.seats[t.winner]) || `Player ${t.winner}`;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${escapeHtml(t.file)}</td><td>${t.turns ?? "—"}</td>` +
      `<td>${escapeHtml(winnerLabel)}</td><td>${escapeHtml((t.seats || []).join(", "))}</td>`;
    tr.addEventListener("click", () => {
      currentTraceFile = t.file;
      loadTrace(run, t.file);
    });
    tbody.appendChild(tr);
  });
}

async function loadTrace(run, file) {
  stopPlay();
  const data = await getJSON(`/api/trace/${encodeURIComponent(run)}/${encodeURIComponent(file)}`);
  initReplay(data);
  showView("view-replay");
}

document.getElementById("back-btn").addEventListener("click", () => {
  if (!document.getElementById("view-replay").hidden) {
    stopPlay();
    loadTraces(currentRun);
  } else if (!document.getElementById("view-traces").hidden) {
    loadRuns();
  }
});

/* ---------------------------------------------------------------------- */
/* Replay: setup                                                         */
/* ---------------------------------------------------------------------- */

function initReplay(data) {
  trace = data;
  header = data.header;
  geo = header.geometry;
  board = header.board;
  plies = data.plies || [];

  const meta = header.meta || {};
  const nPlayers = (plies[0] && plies[0].state.players.length) || 4;
  seatNames = meta.seats || Array.from({ length: nPlayers }, (_, i) => `Player ${i}`);

  computeRollEvents();
  buildBoard();
  buildLog();

  const slider = document.getElementById("ply-slider");
  slider.max = Math.max(0, plies.length - 1);
  slider.value = 0;

  renderPly(0);
}

function computeRollEvents() {
  rollEvents = [];
  plies.forEach((p) => {
    if (p.action_str && p.action_str.startsWith("ROLL") && p.dice) {
      rollEvents.push({ ply: p.ply, dice: p.dice, sum: p.dice[0] + p.dice[1] });
    }
  });
}

/* ---------------------------------------------------------------------- */
/* Board SVG (static layer built once per trace)                         */
/* ---------------------------------------------------------------------- */

function tx(x) { return x; }
function ty(y) { return -y; } // flip so "north" (larger y) renders toward the top

function buildBoard() {
  const svg = document.getElementById("board-svg");
  svg.innerHTML = "";

  const vp = geo.vertex_positions;
  const xs = vp.map((p) => tx(p[0]));
  const ys = vp.map((p) => ty(p[1]));
  const margin = 1.1;
  const minX = Math.min(...xs) - margin;
  const minY = Math.min(...ys) - margin;
  const w = Math.max(...xs) - Math.min(...xs) + margin * 2;
  const h = Math.max(...ys) - Math.min(...ys) + margin * 2;
  svg.setAttribute("viewBox", `${minX} ${minY} ${w} ${h}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  const staticLayer = svgEl("g", { id: "static-layer" });
  svg.appendChild(staticLayer);

  for (let hIdx = 0; hIdx < geo.hex_to_vertices.length; hIdx++) {
    const vids = geo.hex_to_vertices[hIdx];
    const pts = vids.map((v) => `${tx(vp[v][0])},${ty(vp[v][1])}`).join(" ");
    const hexType = board.hex_resources[hIdx];
    const poly = svgEl("polygon", {
      points: pts,
      class: "hex-poly",
      style: `fill:var(${HEX_VAR[hexType]})`,
    });
    staticLayer.appendChild(poly);

    const token = board.hex_tokens[hIdx];
    if (token) {
      const [cx, cy] = geo.hex_centers[hIdx];
      const circle = svgEl("circle", {
        cx: tx(cx), cy: ty(cy), r: 0.32, class: "token-circle",
      });
      staticLayer.appendChild(circle);
      const text = svgEl("text", {
        x: tx(cx), y: ty(cy),
        class: "token-text" + (token === 6 || token === 8 ? " hot" : ""),
      });
      text.textContent = String(token);
      staticLayer.appendChild(text);
    }
  }

  (board.ports || []).forEach((port) => {
    const [va, vb] = port.vertices;
    const pa = vp[va], pb = vp[vb];

    // Port edge midpoint (on the actual edge between vertices)
    const mx = (pa[0] + pb[0]) / 2, my = (pa[1] + pb[1]) / 2;

    // Calculate edge vector
    const ex = pb[0] - pa[0];
    const ey = pb[1] - pa[1];

    // Normal perpendicular to edge (90 degree rotation)
    let nx = -ey;
    let ny = ex;

    // Normalize the normal vector
    const nlen = Math.sqrt(nx * nx + ny * ny);
    nx /= nlen;
    ny /= nlen;

    // Find board center to determine outward direction
    const centerX = geo.hex_centers.reduce((sum, c) => sum + c[0], 0) / geo.hex_centers.length;
    const centerY = geo.hex_centers.reduce((sum, c) => sum + c[1], 0) / geo.hex_centers.length;

    // Check if normal points away from center; flip if needed
    const toEdge_x = mx - centerX;
    const toEdge_y = my - centerY;
    if (nx * toEdge_x + ny * toEdge_y < 0) {
      nx = -nx;
      ny = -ny;
    }

    // Place port label OFF the board (smaller offset to keep in view)
    const offset = 0.4;
    const px = mx + nx * offset;
    const py = my + ny * offset;

    // Draw dashed line from vertex va to port label (neutral color for all)
    const line1 = svgEl("line", {
      x1: tx(pa[0]), y1: ty(pa[1]),
      x2: tx(px), y2: ty(py),
      class: "port-connector",
    });
    staticLayer.appendChild(line1);

    // Draw dashed line from vertex vb to port label (neutral color for all)
    const line2 = svgEl("line", {
      x1: tx(pb[0]), y1: ty(pb[1]),
      x2: tx(px), y2: ty(py),
      class: "port-connector",
    });
    staticLayer.appendChild(line2);

    // Draw port label off the board
    const text = svgEl("text", {
      x: tx(px), y: ty(py),
      class: "port-label",
    });
    text.textContent = port.resource === null || port.resource === undefined
      ? "3:1"
      : RESOURCE_ABBR[port.resource];

    // Add tooltip showing full resource name
    const fullLabel = port.resource === null || port.resource === undefined
      ? "3:1 generic"
      : `${RESOURCE_NAMES[port.resource]} 2:1`;
    const title = svgEl("title", {});
    title.textContent = fullLabel;
    text.appendChild(title);

    staticLayer.appendChild(text);
  });

  const dynLayer = svgEl("g", { id: "dynamic-layer" });
  svg.appendChild(dynLayer);
}

function pentagonPath(cx, cy, r) {
  const pts = [];
  for (let i = 0; i < 5; i++) {
    const angle = ((-90 + i * 72) * Math.PI) / 180;
    pts.push([cx + r * Math.cos(angle), cy + r * Math.sin(angle)]);
  }
  return `M${pts.map((p) => p.join(",")).join("L")}Z`;
}

function renderDynamicBoard(ply) {
  const dyn = document.getElementById("dynamic-layer");
  dyn.innerHTML = "";
  const state = ply.state;
  const vp = geo.vertex_positions;

  const [rx, ry] = geo.hex_centers[state.robber_hex];
  const robber = svgEl("circle", {
    cx: tx(rx) - 0.3, cy: ty(ry) - 0.3, r: 0.17, class: "robber-marker",
  });
  const title = svgEl("title", {});
  title.textContent = "Robber";
  robber.appendChild(title);
  dyn.appendChild(robber);

  state.players.forEach((p, pid) => {
    const seatVar = SEAT_VAR[pid % SEAT_VAR.length];

    (p.road_vertices || []).forEach((eid) => {
      const [va, vb] = geo.edge_to_vertices[eid];
      const pa = vp[va], pb = vp[vb];
      const line = svgEl("line", {
        x1: tx(pa[0]), y1: ty(pa[1]), x2: tx(pb[0]), y2: ty(pb[1]),
        class: "road-line", style: `stroke:var(${seatVar});stroke-width:0.11`,
      });
      dyn.appendChild(line);
    });

    (p.settlement_vertices || []).forEach((vid) => {
      const [x, y] = vp[vid];
      const s = 0.24;
      const rect = svgEl("rect", {
        x: tx(x) - s / 2, y: ty(y) - s / 2, width: s, height: s,
        class: "settlement-rect", style: `fill:var(${seatVar})`,
      });
      dyn.appendChild(rect);
    });

    (p.city_vertices || []).forEach((vid) => {
      const [x, y] = vp[vid];
      const path = svgEl("path", {
        d: pentagonPath(tx(x), ty(y), 0.32),
        class: "city-shape", style: `fill:var(${seatVar})`,
      });
      dyn.appendChild(path);
    });
  });
}

/* ---------------------------------------------------------------------- */
/* Panels                                                                 */
/* ---------------------------------------------------------------------- */

function renderPlayers(state) {
  const container = document.getElementById("players-table");
  container.innerHTML = "";

  state.players.forEach((p, pid) => {
    const publicVp = p.settlements_built + 2 * p.cities_built;
    const hiddenVp = p.dev_cards[4] + p.dev_cards_new[4] + p.played_dev_cards[4];
    const lrBonus = state.longest_road_holder === pid ? 2 : 0;
    const laBonus = state.largest_army_holder === pid ? 2 : 0;
    const totalVp = publicVp + hiddenVp + lrBonus + laBonus;

    const card = document.createElement("div");
    card.className = "player-card";

    const head = document.createElement("div");
    head.className = "player-card-head";
    head.innerHTML =
      `<span class="seat-swatch" style="background:var(${SEAT_VAR[pid % SEAT_VAR.length]})"></span>` +
      `<span class="player-name">${escapeHtml(seatLabel(pid))}</span>` +
      `<span class="vp-total">${totalVp} VP</span>`;
    card.appendChild(head);

    const resRow = document.createElement("div");
    resRow.className = "res-row";
    resRow.innerHTML = RESOURCE_NAMES.map((name, ri) =>
      `<span class="res-chip"><span class="res-dot" style="background:var(${RES_VAR[ri]})"></span>` +
      `${name[0].toUpperCase()}:${p.resources[ri]}</span>`
    ).join("");
    card.appendChild(resRow);

    const devRow = document.createElement("div");
    devRow.className = "dev-row";
    const devChips = DEV_ABBR.map((abbr, di) => {
      const held = p.dev_cards[di], nw = p.dev_cards_new[di], played = p.played_dev_cards[di];
      if (!held && !nw && !played) return "";
      return `<span class="dev-chip">${abbr} ${held}h/${nw}n/${played}p</span>`;
    }).join("");
    devRow.innerHTML = devChips || '<span class="dev-chip" style="opacity:.5">no dev cards</span>';
    card.appendChild(devRow);

    if (state.longest_road_holder === pid || state.largest_army_holder === pid) {
      const badges = document.createElement("div");
      if (state.longest_road_holder === pid) badges.innerHTML += '<span class="badge">Longest Road</span>';
      if (state.largest_army_holder === pid) badges.innerHTML += '<span class="badge">Largest Army</span>';
      card.appendChild(badges);
    }

    const breakdown = document.createElement("div");
    breakdown.className = "vp-breakdown";
    breakdown.textContent =
      `public ${publicVp} + hidden ${hiddenVp}` +
      (lrBonus ? ` + road ${lrBonus}` : "") +
      (laBonus ? ` + army ${laBonus}` : "");
    card.appendChild(breakdown);

    container.appendChild(card);
  });
}

function renderBank(state) {
  const row = document.getElementById("bank-row");
  row.innerHTML = RESOURCE_NAMES.map((name, ri) =>
    `<span class="res-chip"><span class="res-dot" style="background:var(${RES_VAR[ri]})"></span>` +
    `${name}: ${state.bank[ri]}</span>`
  ).join("") + `<span class="res-chip">Dev deck: ${state.dev_deck.length}</span>`;
}

function renderDice(idx) {
  const strip = document.getElementById("dice-strip");
  strip.innerHTML = "";
  const hist = new Array(13).fill(0);

  rollEvents.forEach((ev) => {
    if (ev.ply <= idx) {
      hist[ev.sum]++;
      const chip = document.createElement("span");
      chip.className = "die-chip";
      chip.textContent = `${ev.dice[0]}+${ev.dice[1]}`;
      strip.appendChild(chip);
    }
  });
  if (strip.lastElementChild) strip.lastElementChild.classList.add("current");
  strip.scrollLeft = strip.scrollWidth;

  const histDiv = document.getElementById("dice-histogram");
  histDiv.innerHTML = "";
  const maxCount = Math.max(1, ...hist.slice(2));
  for (let s = 2; s <= 12; s++) {
    const wrap = document.createElement("div");
    wrap.className = "hist-bar-wrap";
    const bar = document.createElement("div");
    bar.className = "hist-bar";
    bar.style.height = `${(hist[s] / maxCount) * 100}%`;
    bar.title = `${s}: ${hist[s]}`;
    wrap.appendChild(bar);
    const label = document.createElement("div");
    label.className = "hist-label";
    label.textContent = String(s);
    wrap.appendChild(label);
    histDiv.appendChild(wrap);
  }
}

function renderReadout(ply) {
  document.getElementById("ply-readout").textContent =
    `ply ${ply.ply}/${plies.length - 1} · turn ${ply.turn} · phase ${ply.phase} · player ${seatLabel(ply.player)}`;
}

function seatChipHtml(pid) {
  const seatVar = SEAT_VAR[pid % SEAT_VAR.length];
  return `<span class="seat-chip"><span class="seat-swatch" style="background:var(${seatVar})"></span>` +
    `${escapeHtml(seatLabel(pid))}</span>`;
}

function resChipHtml(ri) {
  return `<span class="res-chip"><span class="res-dot" style="background:var(${RES_VAR[ri]})"></span>` +
    `${escapeHtml(RESOURCE_NAMES[ri])}</span>`;
}

function renderTradeBanner(state) {
  const banner = document.getElementById("trade-banner");
  const trade = state.pending_trade;
  if (!trade) {
    banner.hidden = true;
    banner.innerHTML = "";
    return;
  }
  banner.hidden = false;

  const offerHtml =
    `<span class="trade-offer-line">${seatChipHtml(trade.proposer)} offers ` +
    `${escapeHtml(String(trade.give_n))}&times; ${resChipHtml(trade.give)} for ` +
    `1&times; ${resChipHtml(trade.get)}</span>`;

  const respChips = state.players.map((_, pid) => {
    if (pid === trade.proposer) return "";
    const resp = trade.responses[String(pid)];
    let status = "pending";
    if (resp === true) status = "accepted";
    else if (resp === false) status = "declined";
    return `<span class="resp-chip resp-${status}">${seatChipHtml(pid)} ` +
      `<span class="resp-status">${escapeHtml(status)}</span></span>`;
  }).join("");

  banner.innerHTML = offerHtml + `<span class="trade-responses">${respChips}</span>`;
}

/* ---------------------------------------------------------------------- */
/* Action log                                                             */
/* ---------------------------------------------------------------------- */

function buildLog() {
  const log = document.getElementById("action-log");
  log.innerHTML = "";
  plies.forEach((p, idx) => {
    const row = document.createElement("div");
    row.className = "log-row";
    row.dataset.ply = String(idx);
    row.innerHTML =
      `<span class="log-ply">${idx}</span>` +
      `<span class="log-player">${escapeHtml(seatLabel(p.player))}</span>` +
      `<span class="log-action">${escapeHtml(p.action_str)}</span>`;
    row.addEventListener("click", () => {
      stopPlay();
      renderPly(idx);
    });
    log.appendChild(row);
  });
}

function highlightLogRow(idx) {
  const log = document.getElementById("action-log");
  const prev = log.querySelector(".log-row.current");
  if (prev) prev.classList.remove("current");
  const row = log.children[idx];
  if (row) {
    row.classList.add("current");
    row.scrollIntoView({ block: "nearest" });
  }
}

/* ---------------------------------------------------------------------- */
/* Ply navigation                                                         */
/* ---------------------------------------------------------------------- */

function renderPly(i) {
  if (!plies.length) return;
  currentPly = Math.max(0, Math.min(plies.length - 1, i));
  document.getElementById("ply-slider").value = String(currentPly);
  const ply = plies[currentPly];
  renderDynamicBoard(ply);
  renderPlayers(ply.state);
  renderBank(ply.state);
  renderTradeBanner(ply.state);
  renderDice(currentPly);
  renderReadout(ply);
  highlightLogRow(currentPly);
}

function jumpTurn(delta) {
  stopPlay();
  if (!plies.length) return;
  const curTurn = plies[currentPly].turn;
  let i = currentPly;
  if (delta > 0) {
    while (i < plies.length - 1 && plies[i].turn === curTurn) i++;
  } else {
    while (i > 0 && plies[i].turn === curTurn) i--;
    if (i > 0) {
      const newTurn = plies[i].turn;
      while (i > 0 && plies[i - 1].turn === newTurn) i--;
    }
  }
  renderPly(i);
}

function stopPlay() {
  if (playing) {
    playing = false;
    clearInterval(playTimer);
    document.getElementById("btn-play").textContent = "▶";
  }
}

function togglePlay() {
  playing = !playing;
  const btn = document.getElementById("btn-play");
  if (playing) {
    btn.textContent = "⏸";
    playTimer = setInterval(() => {
      if (currentPly >= plies.length - 1) {
        stopPlay();
        return;
      }
      renderPly(currentPly + 1);
    }, PLY_MS);
  } else {
    clearInterval(playTimer);
    btn.textContent = "▶";
  }
}

/* ---------------------------------------------------------------------- */
/* Controls wiring                                                        */
/* ---------------------------------------------------------------------- */

document.getElementById("ply-slider").addEventListener("input", (e) => {
  stopPlay();
  renderPly(parseInt(e.target.value, 10));
});
document.getElementById("btn-step-back").addEventListener("click", () => { stopPlay(); renderPly(currentPly - 1); });
document.getElementById("btn-step-fwd").addEventListener("click", () => { stopPlay(); renderPly(currentPly + 1); });
document.getElementById("btn-turn-back").addEventListener("click", () => jumpTurn(-1));
document.getElementById("btn-turn-fwd").addEventListener("click", () => jumpTurn(1));
document.getElementById("btn-play").addEventListener("click", togglePlay);

document.addEventListener("keydown", (e) => {
  if (document.getElementById("view-replay").hidden) return;
  if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) {
    e.preventDefault();
  }
  if (e.key === "ArrowLeft") { stopPlay(); renderPly(currentPly - 1); }
  else if (e.key === "ArrowRight") { stopPlay(); renderPly(currentPly + 1); }
  else if (e.key === "ArrowUp") jumpTurn(-1);
  else if (e.key === "ArrowDown") jumpTurn(1);
});

/* ---------------------------------------------------------------------- */
/* Boot                                                                   */
/* ---------------------------------------------------------------------- */

loadRuns();
