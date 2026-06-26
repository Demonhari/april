"use strict";

// --- Token bootstrap -------------------------------------------------------
// The token lives in memory ONLY. It is never written to localStorage,
// sessionStorage, cookies, the page HTML, the console, or any URL we build.
//
// Native desktop (pywebview): the token is fetched asynchronously through the
// minimal JS bridge, window.pywebview.api.get_token(), only after the
// `pywebviewready` event. The SPA never assumes a synchronously-injected global.
//
// Browser-served desktop: the existing safe flow delivers the token in the URL
// fragment (#token=...). Fragments are never sent to the server; we read it once
// and strip it from the address bar/history immediately.
//
// The actual acquisition logic lives in the DOM-free token_bridge.js module
// (window.AprilDesktopAuth). Pure formatting/redaction lives in
// dashboard_helpers.js (window.AprilDashboard). No authenticated request is made
// until acquireToken() has succeeded — see boot() at the bottom of this file.
let TOKEN = "";

const D = window.AprilDashboard;
const BASE = window.location.origin;
const CONVERSATION_ID = (crypto.randomUUID && crypto.randomUUID()) ||
  String(Date.now()) + "-" + Math.random().toString(16).slice(2);
const DEFAULT_SCREEN = "dashboard";

let selectedProject = null; // {id, name, path} or null

// --- Client-side dashboard state ------------------------------------------
// Last-known-good snapshot. Polls update this; a failed poll leaves the prior
// value intact and flips `online` so the UI degrades instead of blanking.
const state = {
  online: false,
  health: null,
  models: null,
  approvals: [],
  reminders: [],
  tasks: [],
  activity: [],
  briefing: null,
  activeAgent: null,
  lastDecision: null,
  lastRiskLevel: null,
  lastPermissionLevel: null,
  lastRoute: null,
  lastModelId: null,
  pendingApprovalId: null,
};

function patchState(patch) {
  Object.assign(state, patch);
}

// --- DOM helpers -----------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const screenEl = $("#screen");
const bannerEl = $("#banner");
const bannerText = $("#banner-text");

function esc(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function el(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function showBanner(message) {
  bannerText.textContent = message;
  bannerEl.classList.add("show");
}
function showBannerInfo(message) {
  bannerText.textContent = message;
  bannerEl.classList.add("show");
  setTimeout(clearBanner, 2500);
}
function clearBanner() {
  bannerEl.classList.remove("show");
  bannerText.textContent = "";
}
$("#banner-dismiss").addEventListener("click", clearBanner);

function authHeaders(extra) {
  return Object.assign({ Authorization: "Bearer " + TOKEN }, extra || {});
}

// --- API helpers -----------------------------------------------------------
// Loud client for user-initiated actions: surfaces failures via the banner.
async function api(method, path, body) {
  let res;
  try {
    res = await fetch(BASE + path, {
      method,
      headers: authHeaders(body !== undefined ? { "Content-Type": "application/json" } : {}),
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    showBanner("Network error: cannot reach the local API at " + BASE + ". Is `run april` running?");
    throw err;
  }
  if (res.status === 401 || res.status === 403) {
    showBanner("Authentication failed (" + res.status + "). Re-open APRIL Desktop via `run april desktop` to refresh the token.");
    throw new Error("auth-" + res.status);
  }
  if (!res.ok) {
    let detail = res.status + "";
    try {
      const data = await res.json();
      detail = (data && data.error && data.error.message) || JSON.stringify(data);
    } catch (_) { /* keep status */ }
    showBanner("Request failed: " + method + " " + path + " — " + detail);
    throw new Error("http-" + res.status);
  }
  clearBanner();
  if (res.status === 204) return null;
  const text = await res.text();
  return text ? JSON.parse(text) : null;
}

// Silent client for background polling: never spams the banner. Returns null on
// any failure and marks the session offline so the UI shows a degraded state
// while keeping the last-known data.
async function pollGet(path) {
  try {
    const res = await fetch(BASE + path, { headers: authHeaders() });
    if (!res.ok) {
      state.online = false;
      return null;
    }
    const text = await res.text();
    state.online = true;
    return text ? JSON.parse(text) : null;
  } catch (_) {
    state.online = false;
    return null;
  }
}

// --- Top system rail (always present) -------------------------------------
function overallSystem() {
  if (!state.health) return { word: state.online ? "unknown" : "offline", kind: state.online ? "neutral" : "bad" };
  const api = D.statusWord(state.health.status);
  const runtime = state.health.runtime ? D.statusWord(state.health.runtime.status) : "unknown";
  if (D.statusKind(api) === "ok" && D.statusKind(runtime) === "ok") return { word: "online", kind: "ok" };
  if (D.statusKind(api) === "bad" || D.statusKind(runtime) === "bad") return { word: "degraded", kind: "bad" };
  return { word: "degraded", kind: "warn" };
}

function setRailItem(id, dotKind, label, value) {
  const node = document.getElementById(id);
  if (!node) return;
  const dot = node.querySelector(".dot");
  if (dot) dot.className = "dot " + (dotKind || "neutral");
  const lab = node.querySelector(".rail-label");
  if (lab && label !== undefined) lab.textContent = label;
  const val = node.querySelector(".rail-val");
  if (val) val.textContent = value;
}

function updateRail() {
  const sys = overallSystem();
  setRailItem("rail-systems", sys.kind, "systems", sys.word);
  const backend = D.backendInfo(state.health);
  const badge = backend.badge === "SIMULATED" ? "simulated" : backend.badge === "REAL" ? "real" : "unknown";
  setRailItem("rail-backend", undefined, "backend", backend.backend + " · " + badge);
  setRailItem("rail-project", undefined, "project", selectedProject ? selectedProject.name : "no project");
  setRailItem("rail-conv", undefined, "conv", CONVERSATION_ID.slice(0, 8));
}

// Back-compat name: refreshes health + the always-on rail. Called only from the
// polling loop, which starts strictly after token acquisition succeeds.
async function refreshConnection() {
  const data = await pollGet("/health");
  if (data) state.health = data;
  updateRail();
  renderDashboard();
}

// --- Screen registry -------------------------------------------------------
const screens = {};
let activeScreen = DEFAULT_SCREEN;

function setActive(name) {
  activeScreen = name;
  document.querySelectorAll("#nav .navitem").forEach((b) => {
    b.classList.toggle("active", b.dataset.screen === name);
  });
}

async function navigate(name) {
  if (!screens[name]) name = DEFAULT_SCREEN;
  setActive(name);
  screenEl.innerHTML = "";
  try {
    await screens[name]();
  } catch (err) {
    if (!String(err.message || "").startsWith("auth-") && !String(err.message || "").startsWith("http-")) {
      console.error(err);
    }
  }
}

function bindNav() {
  document.querySelectorAll("#nav .navitem").forEach((b) => {
    b.addEventListener("click", () => navigate(b.dataset.screen));
  });
}

function screenHeader(title, sub) {
  const h = el("div");
  h.innerHTML = "<h1 class='screen-title'>" + esc(title) + "</h1><p class='screen-sub'>" + esc(sub || "") + "</p>";
  return h;
}
function card(html) {
  return el("div", "card", html);
}
function panel(title, bodyId, rightHtml) {
  const p = el("div", "panel");
  p.innerHTML =
    "<div class='panel-title'>" + esc(title) +
    (rightHtml ? "<span class='right'>" + rightHtml + "</span>" : "") +
    "</div><div class='panel-body' id='" + esc(bodyId) + "'></div>";
  return p;
}

// ===========================================================================
//  DASHBOARD (default cockpit screen)
// ===========================================================================
screens.dashboard = async function () {
  const cockpit = el("div", "cockpit");

  const left = el("div", "col-left");
  left.appendChild(panel("Systems", "dash-systems"));

  const center = el("div", "col-center");
  center.appendChild(panel("Agent Router", "dash-orbit"));
  center.appendChild(panel("Active Specialist", "dash-specialist"));
  center.appendChild(panel("Permission Level", "dash-permission"));
  center.appendChild(panel("Runtime Telemetry", "dash-telemetry"));
  center.appendChild(panel("Runtime Models", "dash-models"));

  const right = el("div", "col-right");
  right.appendChild(panel("Approvals", "dash-approvals"));
  right.appendChild(panel("Reminders & Tasks", "dash-reminders"));

  const feed = el("div", "cockpit-feed");
  feed.appendChild(panel("Activity Feed · redacted", "dash-feed"));
  const consolePanel = panel("Command Console", "dash-console");
  feed.appendChild(consolePanel);

  const command = el("div", "cockpit-command");
  command.appendChild(buildCommandBar());

  cockpit.appendChild(left);
  cockpit.appendChild(center);
  cockpit.appendChild(right);
  cockpit.appendChild(feed);
  cockpit.appendChild(command);
  screenEl.appendChild(cockpit);

  renderDashboard();
};

// Re-render all poll-driven regions from `state`. No-op for any region whose
// container is absent (e.g. when a detail screen is mounted). NEVER touches the
// command bar or the console, which hold live user input / chat output.
function renderDashboard() {
  renderSystems();
  renderOrbit();
  renderSpecialist();
  renderPermission();
  renderTelemetry();
  renderModels();
  renderApprovals();
  renderReminders();
  renderFeed();
}

function renderInto(id, build) {
  const host = document.getElementById(id);
  if (!host) return;
  host.innerHTML = "";
  build(host);
}

function sysRow(name, value, kind) {
  return "<div class='sys-row'><span class='dot " + (kind || "neutral") + "'></span>" +
    "<span class='sys-name'>" + esc(name) + "</span>" +
    "<span class='sys-val'>" + esc(value) + "</span></div>";
}

function renderSystems() {
  renderInto("dash-systems", (host) => {
    if (!state.health) {
      host.innerHTML = "<span class='muted'>" + (state.online ? "loading…" : "offline — last status unavailable") + "</span>";
      return;
    }
    const s = D.subsystems(state.health);
    host.innerHTML =
      sysRow("Core API", s.core_api, D.statusKind(s.core_api)) +
      sysRow("April Runtime", s.runtime, D.statusKind(s.runtime)) +
      sysRow("Backend", s.backend, "neutral") +
      sysRow("Database", s.database, D.statusKind(s.database)) +
      sysRow("Vector index", s.vector_index, D.statusKind(s.vector_index)) +
      sysRow("Voice", s.voice, D.statusKind(s.voice)) +
      sysRow("Scheduler", s.scheduler, D.statusKind(s.scheduler));
  });
}

function renderOrbit() {
  renderInto("dash-orbit", (host) => {
    const wrap = el("div", "orbit-wrap");
    const orbit = el("div", "orbit");
    orbit.appendChild(el("div", "orbit-ring"));
    orbit.appendChild(el("div", "orbit-ring inner"));
    const activeEntry = D.agentEntry(state.activeAgent);
    const coreSub = activeEntry ? activeEntry.code : (state.online ? "idle" : "offline");
    orbit.appendChild(el("div", "orbit-core",
      "<span class='core-label'>ROUTER</span><span class='core-sub'>" + esc(coreSub) + "</span>"));

    const size = 230;
    const radius = 92;
    D.AGENTS.forEach((agent, i) => {
      const angle = (-90 + i * (360 / D.AGENTS.length)) * (Math.PI / 180);
      const x = size / 2 + radius * Math.cos(angle);
      const y = size / 2 + radius * Math.sin(angle);
      const isActive = activeEntry && activeEntry.id === agent.id;
      const node = el("div", "orbit-node" + (isActive ? " active" : ""),
        "<span class='node-code'>" + esc(agent.code) + "</span>");
      node.style.left = x + "px";
      node.style.top = y + "px";
      node.title = agent.label + " agent (" + agent.id + ")";
      orbit.appendChild(node);
    });
    wrap.appendChild(orbit);

    const legend = el("div", "orbit-legend");
    legend.innerHTML = D.AGENTS.map((a) => {
      const on = activeEntry && activeEntry.id === a.id;
      return "<span class='leg" + (on ? " active" : "") + "'>" + esc(a.code) + " " + esc(a.label) + "</span>";
    }).join("");
    wrap.appendChild(legend);

    const status = activeEntry
      ? "active: " + esc(activeEntry.label) + " agent"
      : "active agent: unknown — send a message to route";
    wrap.appendChild(el("div", "kv", "<div class='spacer'></div>" + status));
    host.appendChild(wrap);
  });
}

function valOrUnknown(value) {
  return D.isMissing(value) ? "unknown" : esc(value);
}

function renderSpecialist() {
  renderInto("dash-specialist", (host) => {
    const entry = D.agentEntry(state.activeAgent);
    const name = entry ? entry.label + " agent" : (state.activeAgent ? state.activeAgent : "unknown");
    const code = entry ? entry.code : "—";
    const current = D.currentPermissionLevel(state.lastPermissionLevel, state.approvals);
    host.innerHTML =
      "<div class='row'><span class='pill cyan'>" + esc(code) + "</span>" +
      "<strong>" + esc(name) + "</strong></div><div class='spacer'></div>" +
      "<div class='kv'><strong>model</strong> " + valOrUnknown(state.lastModelId) + "</div>" +
      "<div class='kv'><strong>status</strong> " + valOrUnknown(state.lastDecision) + "</div>" +
      "<div class='kv'><strong>route</strong> " + valOrUnknown(state.lastRoute) + "</div>" +
      "<div class='kv'><strong>risk</strong> " + valOrUnknown(state.lastRiskLevel) + "</div>" +
      "<div class='kv'><strong>permission</strong> " + (current === null ? "unknown" : "Level " + current) + "</div>";
  });
}

function renderPermission() {
  renderInto("dash-permission", (host) => {
    const current = D.currentPermissionLevel(state.lastPermissionLevel, state.approvals);
    const hasPending = (state.approvals && state.approvals.length > 0) || !!state.pendingApprovalId;
    const ladder = el("div", "levels");
    ladder.innerHTML = D.PERMISSION_LEVELS.map((lv) =>
      "<div class='level" + (current === lv ? " current" : "") + "'>" +
      "<div class='lv-num'>" + lv + "</div>L" + lv + "</div>").join("");
    host.appendChild(ladder);
    const note = el("div", "kv");
    note.innerHTML = "<div class='spacer'></div>" +
      (current === null ? "No elevated permission required yet." : "Last required: <strong>Level " + current + "</strong>.") +
      " Level 3+ tools need exact-ID approval." +
      (hasPending ? " <span class='pill warn'>pending exact approval</span>" : "");
    host.appendChild(note);
  });
}

function metric(label, value, isUnknown) {
  return "<div class='metric'><div class='m-label'>" + esc(label) + "</div>" +
    "<div class='m-val" + (isUnknown ? " unknown" : "") + "'>" + esc(value) + "</div></div>";
}

function renderTelemetry() {
  renderInto("dash-telemetry", (host) => {
    const t = D.telemetryFrom(state.health, state.models && state.models.models);
    const mk = (label, raw, fmt) => {
      const unknown = D.isMissing(raw);
      return metric(label, unknown ? "unknown" : fmt(raw), unknown);
    };
    const grid = el("div", "metrics");
    grid.innerHTML =
      mk("tokens/sec", t.tokens_per_second, (v) => D.formatRate(v)) +
      mk("first-token", t.first_token_latency_ms, (v) => D.formatRate(v, " ms")) +
      mk("context", t.context_size, (v) => D.formatInt(v)) +
      mk("process RSS", t.process_rss_bytes, (v) => D.formatBytes(v)) +
      mk("loaded models", t.loaded_model_count, (v) => D.formatInt(v)) +
      mk("active reqs", t.active_requests, (v) => D.formatInt(v)) +
      mk("gen errors", t.generation_error_count, (v) => D.formatInt(v));
    host.appendChild(grid);
  });
}

function modelLoadButtons(modelId) {
  const row = el("div", "row");
  row.innerHTML =
    "<button class='btn secondary' data-act='load'>Load</button>" +
    "<button class='btn secondary' data-act='unload'>Unload</button>";
  row.querySelector("[data-act='load']").addEventListener("click", async () => {
    await api("POST", "/runtime/models/load", { model_id: modelId });
    showBannerInfo("Requested load of " + modelId + ".");
    await refreshModels();
  });
  row.querySelector("[data-act='unload']").addEventListener("click", async () => {
    await api("POST", "/runtime/models/unload", { model_id: modelId });
    showBannerInfo("Requested unload of " + modelId + ".");
    await refreshModels();
  });
  return row;
}

function renderModels() {
  renderInto("dash-models", (host) => {
    const backend = D.backendInfo(state.health);
    if (!state.models) {
      host.innerHTML = "<span class='muted'>" + (state.online ? "loading…" : "not available") + "</span>";
      return;
    }
    const models = (state.models && state.models.models) || [];
    if (!models.length) {
      host.innerHTML = "<span class='muted'>No models reported.</span>";
      return;
    }
    models.forEach((m) => {
      const loaded = m.state === "loaded";
      const item = el("div", "list-item");
      // NOTE: m.path is intentionally never rendered (unredacted model path).
      item.innerHTML =
        "<div class='row'><strong>" + esc(m.id) + "</strong>" +
        "<span class='pill'>" + esc(m.role) + "</span>" +
        "<span class='pill " + (loaded ? "ok" : "") + "'>" + esc(m.state) + "</span>" +
        (m.keep_loaded ? "<span class='pill cyan'>keep_loaded</span>" : "") +
        (m.missing_path ? "<span class='pill warn'>missing path</span>" : "") +
        (backend.simulated === true ? "<span class='pill warn'>simulated</span>" : "") +
        "</div>";
      item.appendChild(modelLoadButtons(m.id));
      host.appendChild(item);
    });
  });
}

function approvalItem(ap, onChange) {
  const item = el("div", "list-item");
  const meta = ap.metadata || {};
  const digest = ap.canonical_hash || meta.artifact_sha256 || meta.digest || meta.manifest_sha256 || "";
  item.innerHTML =
    "<div class='row'><span class='pill warn'>" + esc(ap.tool) + "</span>" +
    "<span class='pill'>risk " + esc(ap.risk_level) + "</span>" +
    "<span class='pill'>level " + esc(ap.permission_level) + "</span>" +
    "<span class='grow'></span><span class='kv'>expires " + esc(ap.expires_at) + "</span></div>" +
    "<div class='kv'><strong>approval_id</strong> " + esc(ap.id) + "</div>" +
    (digest ? "<div class='kv'><strong>digest</strong> " + esc(digest) + "</div>" : "") +
    "<div class='spacer'></div><div class='row'>" +
    "<button class='btn approve' data-act='approve'>Approve this exact ID</button>" +
    "<button class='btn danger' data-act='deny'>Deny</button></div>";
  item.querySelector("[data-act='approve']").addEventListener("click", async () => {
    await api("POST", "/tools/approve", { approval_id: ap.id });
    showBannerInfo("Approved " + ap.id + ".");
    if (state.pendingApprovalId === ap.id) state.pendingApprovalId = null;
    await onChange();
  });
  item.querySelector("[data-act='deny']").addEventListener("click", async () => {
    await api("POST", "/tools/deny", { approval_id: ap.id });
    showBannerInfo("Denied " + ap.id + ".");
    if (state.pendingApprovalId === ap.id) state.pendingApprovalId = null;
    await onChange();
  });
  return item;
}

function renderApprovals() {
  const host = document.getElementById("dash-approvals");
  if (!host) return;
  host.innerHTML = "";
  const approvals = state.approvals || [];
  if (!approvals.length) {
    host.innerHTML = "<span class='muted'>Nothing awaiting approval. A chat 'yes' never approves.</span>";
    return;
  }
  approvals.forEach((ap) => host.appendChild(approvalItem(ap, async () => {
    await refreshApprovals();
  })));
}

function renderReminders() {
  renderInto("dash-reminders", (host) => {
    const reminders = state.reminders || [];
    const tasks = state.tasks || [];
    const next = reminders.find((r) => r.due_at && !r.fired_at) || reminders[0];
    let html =
      "<div class='row'><span class='pill cyan'>" + reminders.length + " reminders</span>" +
      "<span class='pill cyan'>" + tasks.length + " tasks</span></div><div class='spacer'></div>";
    html += "<div class='kv'><strong>next</strong> " +
      (next ? esc(next.content) + (next.due_at ? " · due " + esc(next.due_at) : " · no due date") : "none") + "</div>";
    if (state.briefing) {
      html += "<div class='spacer'></div><div class='kv'><strong>briefing</strong> " +
        esc(state.briefing.title || "Today's briefing") + "</div>" +
        "<pre>" + esc((state.briefing.body || "").slice(0, 600)) + "</pre>";
    } else {
      html += "<div class='spacer'></div><div class='kv'>briefing: not available</div>";
    }
    host.innerHTML = html;
  });
}

function renderFeed() {
  renderInto("dash-feed", (host) => {
    const events = state.activity || [];
    if (!events.length) {
      host.innerHTML = "<span class='muted'>" + (state.online ? "No audit events yet." : "offline") + "</span>";
      return;
    }
    const feed = el("div", "feed");
    events.slice(0, 60).forEach((e) => {
      const r = D.activityRow(e); // client-side allowlist projection (defence in depth)
      const bits = [];
      if (r.agent) bits.push("agent " + esc(r.agent));
      if (r.tool) bits.push("tool " + esc(r.tool));
      if (r.permission !== null) bits.push("L" + r.permission);
      if (r.outcome) bits.push(esc(r.outcome));
      if (r.ref) bits.push("#" + esc(r.ref));
      feed.appendChild(el("div", "feed-row",
        "<span class='feed-time'>" + esc(r.time) + "</span>" +
        "<span class='feed-body'><span class='feed-kind'>" + esc(r.kind) + "</span>" +
        (r.risk ? "<span class='feed-risk'>risk " + esc(r.risk) + "</span>" : "") +
        "<span class='feed-meta'>" + bits.join(" · ") + "</span></span>"));
    });
    host.appendChild(feed);
  });
}

// --- Command bar + chat streaming -----------------------------------------
let chatBusy = false;

function buildCommandBar() {
  const bar = el("div");
  const wrap = el("div", "cmd-bar");
  wrap.innerHTML =
    "<span class='prompt'>&gt;</span>" +
    "<textarea id='cmd-input' placeholder='Ask APRIL… (Enter to send, Shift+Enter for newline)'></textarea>" +
    "<button class='btn' id='cmd-send'>Send</button>";
  bar.appendChild(wrap);
  bar.appendChild(el("div", "cmd-meta",
    "Streams from POST /chat/stream over loopback. Approvals are never granted from chat — use the Approvals panel."));
  // Bind after insertion into the DOM happens by the caller; defer lookups.
  setTimeout(() => {
    const input = document.getElementById("cmd-input");
    const send = document.getElementById("cmd-send");
    if (!input || !send) return;
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submitCommand();
      }
    });
    send.addEventListener("click", submitCommand);
  }, 0);
  return bar;
}

function consoleHost() {
  return document.getElementById("dash-console") || document.getElementById("chat-console");
}

function consoleLine(kind, html) {
  const host = consoleHost();
  if (!host) return null;
  let log = host.querySelector(".console");
  if (!log) {
    log = el("div", "console");
    host.appendChild(log);
  }
  const node = el("div", "msg " + kind, html);
  log.appendChild(node);
  node.scrollIntoView({ block: "end" });
  return node;
}

async function submitCommand() {
  const input = document.getElementById("cmd-input") || document.getElementById("chat-input");
  if (!input) return;
  const message = (input.value || "").trim();
  if (!message || chatBusy) return;
  input.value = "";
  await streamChat(message);
}

async function streamChat(message) {
  chatBusy = true;
  const sendBtn = document.getElementById("cmd-send") || document.getElementById("chat-send");
  if (sendBtn) sendBtn.disabled = true;
  consoleLine("user", esc(message));
  const chips = el("div", "chips");
  const chipHost = consoleHost();
  if (chipHost) {
    const log = chipHost.querySelector(".console");
    if (log) log.appendChild(chips);
  }
  const assistant = consoleLine("assistant", "<span class='muted'>…</span>");
  let assistantText = "";

  try {
    const res = await fetch(BASE + "/chat/stream", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        message,
        conversation_id: CONVERSATION_ID,
        project_id: selectedProject ? selectedProject.id : null,
      }),
    });
    if (res.status === 401 || res.status === 403) {
      showBanner("Authentication failed (" + res.status + "). Re-open via `run april desktop`.");
      if (assistant) { assistant.className = "msg errline"; assistant.textContent = "Authentication failed."; }
      return;
    }
    if (!res.ok || !res.body) {
      if (assistant) { assistant.className = "msg errline"; assistant.textContent = "Stream failed (" + res.status + ")."; }
      return;
    }
    clearBanner();
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const evt = parseSse(frame);
        if (!evt) continue;
        assistantText = handleChatEvent(evt, assistant, assistantText, chips);
      }
    }
  } catch (_err) {
    showBanner("Chat stream network error. Is the local API running?");
    if (assistant) { assistant.className = "msg errline"; assistant.textContent = "Network error during stream."; }
  } finally {
    chatBusy = false;
    if (sendBtn) sendBtn.disabled = false;
    // A finished turn may have produced or consumed an approval; refresh.
    refreshApprovals();
  }
}

function parseSse(frame) {
  const lines = frame.split("\n");
  let dataLine = null;
  for (const line of lines) {
    if (line.startsWith("data:")) dataLine = line.slice(5).trim();
  }
  if (!dataLine) return null;
  try {
    return JSON.parse(dataLine);
  } catch (_) {
    return null;
  }
}

function addChip(chips, text, cls) {
  if (!chips || !text) return;
  chips.appendChild(el("span", "chip" + (cls ? " " + cls : ""), esc(text)));
}

function handleChatEvent(evt, assistant, assistantText, chips) {
  // Update dashboard state from structural fields only (never message text).
  const patch = D.streamStateUpdate(evt);
  if (Object.keys(patch).length) {
    patchState(patch);
    renderOrbit();
    renderSpecialist();
    renderPermission();
  }
  const chip = D.summarizeStreamEvent(evt);
  if (chip) addChip(chips, chip, evt.event === "error" ? "err" : (evt.event === "approval_required" ? "warn" : ""));

  const event = evt.event;
  const payload = evt.payload || {};
  if (event === "token") {
    assistantText += payload.text || "";
    if (assistant) { assistant.textContent = assistantText; assistant.scrollIntoView({ block: "end" }); }
  } else if (event === "approval_required") {
    const ap = payload.approval || {};
    consoleLine("approval",
      "<strong>Approval required.</strong> APRIL paused and will not act without explicit, exact-ID approval." +
      "<div class='spacer'></div><span class='kv'>approval_id: " + esc(ap.id || "") + "</span>" +
      "<div class='spacer'></div><button class='btn' id='goto-approvals'>Review in Approvals</button>");
    const btn = document.getElementById("goto-approvals");
    if (btn) btn.addEventListener("click", () => navigate("approvals"));
    // Route the user's attention to the approval surface on the dashboard.
    refreshApprovals().then(flashApprovals);
  } else if (event === "error") {
    consoleLine("errline", "Error: " + esc(payload.message || payload.code || "generation failed"));
  }
  return assistantText;
}

function flashApprovals() {
  const host = document.getElementById("dash-approvals");
  if (!host) return;
  const card = host.closest(".panel");
  if (!card) return;
  card.scrollIntoView({ block: "nearest" });
  card.style.transition = "box-shadow 0.4s ease";
  card.style.boxShadow = "0 0 18px -2px rgba(255, 165, 59, 0.8)";
  setTimeout(() => { card.style.boxShadow = ""; }, 1600);
}

// ===========================================================================
//  POLLING
// ===========================================================================
let pollTimers = [];

async function refreshHealth() {
  const data = await pollGet("/health");
  if (data) state.health = data;
  updateRail();
  renderSystems();
  renderTelemetry();
  renderOrbit();
}

async function refreshApprovals() {
  const data = await pollGet("/approvals");
  if (data) state.approvals = (data && data.approvals) || [];
  renderApprovals();
  renderPermission();
  renderSpecialist();
}

async function refreshActivity() {
  const data = await pollGet("/diagnostics/activity?limit=80");
  if (data) state.activity = (data && data.events) || [];
  renderFeed();
}

async function refreshModels() {
  const data = await pollGet("/runtime/models");
  if (data) state.models = data;
  renderModels();
  renderTelemetry();
}

async function refreshSlow() {
  const reminders = await pollGet("/reminders");
  if (reminders) state.reminders = (reminders && reminders.reminders) || [];
  const tasks = await pollGet("/tasks");
  if (tasks) state.tasks = (tasks && tasks.tasks) || [];
  const briefing = await pollGet("/scheduler/briefing/preview");
  if (briefing) state.briefing = briefing;
  renderReminders();
}

async function refreshDashboard() {
  await Promise.all([refreshHealth(), refreshApprovals(), refreshActivity(), refreshModels(), refreshSlow()]);
  updateRail();
}

function startPolling() {
  // Initial fill, then staggered cadences. Loopback-only; intentionally modest.
  refreshDashboard();
  pollTimers.push(setInterval(refreshConnection, 8000)); // health + rail
  pollTimers.push(setInterval(refreshApprovals, 8000));
  pollTimers.push(setInterval(refreshActivity, 10000));
  pollTimers.push(setInterval(refreshModels, 13000));
  pollTimers.push(setInterval(refreshSlow, 45000));
}

// ===========================================================================
//  DETAIL SCREENS (kept available; restyled to the cockpit theme)
// ===========================================================================
screens.chat = async function () {
  screenEl.appendChild(screenHeader("Chat", "Streams from POST /chat/stream over loopback. One conversation per session."));
  const projLabel = selectedProject
    ? "Project scope: " + esc(selectedProject.name)
    : "No project selected (general chat). Pick one under Projects.";
  screenEl.appendChild(card(
    "<div class='row'><span class='pill cyan'>" + projLabel + "</span>" +
    "<span class='grow'></span><span class='kv'>conversation " + esc(CONVERSATION_ID.slice(0, 8)) + "</span></div>"));
  const consolePanel = panel("Console", "chat-console");
  screenEl.appendChild(consolePanel);
  const compose = el("div", "cockpit-command");
  compose.appendChild(buildCommandBarChat());
  screenEl.appendChild(compose);
};

function buildCommandBarChat() {
  const bar = el("div");
  const wrap = el("div", "cmd-bar");
  wrap.innerHTML =
    "<span class='prompt'>&gt;</span>" +
    "<textarea id='chat-input' placeholder='Ask APRIL… (Enter to send, Shift+Enter for newline)'></textarea>" +
    "<button class='btn' id='chat-send'>Send</button>";
  bar.appendChild(wrap);
  bar.appendChild(el("div", "cmd-meta", "Approvals are never granted from chat. Use the Approvals screen."));
  setTimeout(() => {
    const input = document.getElementById("chat-input");
    const send = document.getElementById("chat-send");
    if (!input || !send) return;
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submitCommand(); }
    });
    send.addEventListener("click", submitCommand);
    input.focus();
  }, 0);
  return bar;
}

screens.projects = async function () {
  screenEl.appendChild(screenHeader("Projects", "Selecting a project scopes Chat and coding work. Paths must resolve inside allowed roots."));
  const data = await api("GET", "/projects");
  const projects = (data && data.projects) || [];
  const add = card(
    "<div class='row'><input class='grow' id='proj-path' placeholder='/absolute/path/to/repo' />" +
    "<input id='proj-name' placeholder='name (optional)' />" +
    "<button class='btn' id='proj-add'>Add project</button></div>");
  screenEl.appendChild(add);
  $("#proj-add").addEventListener("click", async () => {
    const path = $("#proj-path").value.trim();
    if (!path) return;
    const name = $("#proj-name").value.trim();
    await api("POST", "/projects", { path, name: name || null });
    navigate("projects");
  });
  const list = card("<div class='panel-title'>Registered projects</div>");
  if (!projects.length) list.innerHTML += "<span class='muted'>No projects yet.</span>";
  for (const p of projects) {
    const item = el("div", "list-item");
    const selected = selectedProject && selectedProject.id === p.id;
    item.innerHTML =
      "<div class='row'><strong>" + esc(p.name) + "</strong>" +
      (selected ? " <span class='pill ok'>selected</span>" : "") +
      "<span class='grow'></span>" +
      "<button class='btn secondary' data-act='select'>" + (selected ? "Selected" : "Select") + "</button>" +
      "<button class='btn secondary' data-act='index'>Index</button></div>" +
      "<div class='kv'>" + esc(p.path) + "</div><div class='kv'>id " + esc(p.id) + "</div>";
    item.querySelector("[data-act='select']").addEventListener("click", () => {
      selectedProject = selected ? null : { id: p.id, name: p.name, path: p.path };
      updateRail();
      navigate("projects");
    });
    item.querySelector("[data-act='index']").addEventListener("click", async () => {
      await api("POST", "/projects/" + encodeURIComponent(p.id) + "/index", {});
      showBannerInfo("Indexed project " + p.name + ".");
    });
    list.appendChild(item);
  }
  screenEl.appendChild(list);
};

screens.approvals = async function () {
  screenEl.appendChild(screenHeader("Approvals", "Exact-ID, one-time approvals. A chat 'yes' is never approval."));
  const data = await api("GET", "/approvals");
  state.approvals = (data && data.approvals) || [];
  const wrap = card("<div class='panel-title'>Pending approvals</div>");
  if (!state.approvals.length) wrap.innerHTML += "<span class='muted'>Nothing awaiting approval.</span>";
  for (const ap of state.approvals) {
    wrap.appendChild(approvalItem(ap, async () => { navigate("approvals"); }));
  }
  screenEl.appendChild(wrap);
};

screens.memory = async function () {
  screenEl.appendChild(screenHeader("Memory", "Explicit durable local memory. Nothing is auto-created; you see exactly what is stored."));
  const search = card(
    "<div class='row'><input class='grow' id='mem-q' placeholder='search memories' value='*' />" +
    "<button class='btn' id='mem-search'>Search</button>" +
    "<button class='btn secondary' id='mem-export'>Export all</button></div>");
  screenEl.appendChild(search);
  const create = card(
    "<div class='panel-title'>Store a memory</div>" +
    "<textarea id='mem-content' placeholder='content to remember'></textarea><div class='spacer'></div>" +
    "<div class='row'><select id='mem-type'><option value='fact'>fact</option>" +
    "<option value='preference'>preference</option></select>" +
    "<input class='grow' id='mem-reason' placeholder='reason (why store this)' />" +
    "<button class='btn' id='mem-add'>Store</button></div>");
  screenEl.appendChild(create);
  const results = el("div");
  results.id = "mem-results";
  screenEl.appendChild(results);

  const runSearch = async () => {
    const q = $("#mem-q").value.trim() || "*";
    const data = await api("GET", "/memory/search?q=" + encodeURIComponent(q));
    renderMemoryList((data && data.results) || []);
  };
  $("#mem-search").addEventListener("click", runSearch);
  $("#mem-export").addEventListener("click", async () => {
    const data = await api("GET", "/memory/export");
    results.innerHTML = "";
    results.appendChild(card("<div class='panel-title'>Export</div><pre>" + esc(JSON.stringify(data.export, null, 2)) + "</pre>"));
  });
  $("#mem-add").addEventListener("click", async () => {
    const content = $("#mem-content").value.trim();
    const reason = $("#mem-reason").value.trim();
    if (!content || !reason) {
      showBanner("Both content and reason are required to store a memory.");
      return;
    }
    await api("POST", "/memory", {
      content, reason, memory_type: $("#mem-type").value,
      project_id: selectedProject ? selectedProject.id : null,
    });
    $("#mem-content").value = "";
    $("#mem-reason").value = "";
    showBannerInfo("Memory stored.");
    runSearch();
  });

  function renderMemoryList(items) {
    results.innerHTML = "";
    const wrap = card("<div class='panel-title'>" + items.length + " memor" + (items.length === 1 ? "y" : "ies") + "</div>");
    for (const m of items) {
      const node = el("div", "list-item");
      node.innerHTML =
        "<div class='row'><span class='pill'>" + esc(m.kind) + "</span>" +
        (m.project_id ? "<span class='pill'>project " + esc(String(m.project_id).slice(0, 8)) + "</span>" : "") +
        "<span class='grow'></span><button class='btn danger' data-act='del'>Delete</button></div>" +
        "<div>" + esc(m.content) + "</div>" +
        "<div class='kv'>reason: " + esc(m.reason) + "</div><div class='kv'>id " + esc(m.id) + "</div>";
      node.querySelector("[data-act='del']").addEventListener("click", async () => {
        await api("DELETE", "/memory/" + encodeURIComponent(m.id));
        showBannerInfo("Deleted memory.");
        runSearch();
      });
      wrap.appendChild(node);
    }
    results.appendChild(wrap);
  }
  await runSearch();
};

screens.reminders = async function () {
  screenEl.appendChild(screenHeader("Reminders & Tasks", "Local reminders, inspectable task plans, and today's briefing."));
  const briefingCard = card("<div class='panel-title'>Today's briefing</div><span class='muted'>loading…</span>");
  screenEl.appendChild(briefingCard);
  try {
    const b = await api("GET", "/scheduler/briefing/preview");
    state.briefing = b;
    briefingCard.innerHTML =
      "<div class='panel-title'>Today's briefing</div>" +
      "<div><strong>" + esc(b.title || "Briefing") + "</strong></div><pre>" + esc(b.body || "") + "</pre>";
  } catch (_) {
    briefingCard.innerHTML = "<div class='panel-title'>Today's briefing</div><span class='muted'>unavailable</span>";
  }
  const add = card(
    "<div class='panel-title'>New reminder</div><div class='row'>" +
    "<input class='grow' id='rem-content' placeholder='reminder content' />" +
    "<input id='rem-due' placeholder='due_at ISO (optional)' />" +
    "<button class='btn' id='rem-add'>Add</button></div>");
  screenEl.appendChild(add);
  $("#rem-add").addEventListener("click", async () => {
    const content = $("#rem-content").value.trim();
    if (!content) return;
    const due = $("#rem-due").value.trim();
    await api("POST", "/reminders", { content, due_at: due || null });
    navigate("reminders");
  });
  const remData = await api("GET", "/reminders");
  state.reminders = (remData && remData.reminders) || [];
  const remWrap = card("<div class='panel-title'>Reminders</div>");
  if (!state.reminders.length) remWrap.innerHTML += "<span class='muted'>None.</span>";
  for (const r of state.reminders) {
    const node = el("div", "list-item");
    node.innerHTML =
      "<div class='row'><span class='grow'>" + esc(r.content) + "</span>" +
      "<button class='btn danger' data-act='del'>Delete</button></div>" +
      "<div class='kv'>" + (r.due_at ? "due " + esc(r.due_at) : "no due date") +
      (r.fired_at ? " · fired " + esc(r.fired_at) : "") + " · id " + esc(r.id) + "</div>";
    node.querySelector("[data-act='del']").addEventListener("click", async () => {
      await api("DELETE", "/reminders/" + encodeURIComponent(r.id));
      navigate("reminders");
    });
    remWrap.appendChild(node);
  }
  screenEl.appendChild(remWrap);
  const taskData = await api("GET", "/tasks");
  state.tasks = (taskData && taskData.tasks) || [];
  const taskWrap = card("<div class='panel-title'>Tasks</div>");
  if (!state.tasks.length) taskWrap.innerHTML += "<span class='muted'>No task plans.</span>";
  for (const t of state.tasks) {
    taskWrap.appendChild(el("div", "list-item", "<pre>" + esc(JSON.stringify(t, null, 2)) + "</pre>"));
  }
  screenEl.appendChild(taskWrap);
};

screens.status = async function () {
  screenEl.appendChild(screenHeader("Status & Models", "Local health, diagnostics, and runtime model control."));
  const health = await api("GET", "/health").catch(() => null);
  if (health) {
    state.health = health;
    const backend = D.backendInfo(health);
    const badge = backend.badge === "SIMULATED"
      ? "<span class='pill warn'>SIMULATED runtime (fake backend) — not real-model verified</span>"
      : backend.badge === "REAL" ? "<span class='pill ok'>real backend</span>"
        : "<span class='pill'>backend unknown</span>";
    let note = "<div class='panel-title'>Runtime mode</div>" + badge;
    if (backend.missing_models.length) {
      note += "<div class='spacer'></div><span class='muted'>Configured but missing model files: " +
        esc(backend.missing_models.join(", ")) + "</span>";
    }
    screenEl.appendChild(card(note));
    screenEl.appendChild(card("<div class='panel-title'>Health (redacted /health)</div><pre>" + esc(JSON.stringify(health, null, 2)) + "</pre>"));
  }
  try {
    const diag = await api("GET", "/diagnostics");
    screenEl.appendChild(card("<div class='panel-title'>Diagnostics</div><pre>" + esc(JSON.stringify(diag, null, 2)) + "</pre>"));
  } catch (_) { /* surfaced via banner */ }
  const modelData = await api("GET", "/runtime/models").catch(() => null);
  if (modelData) state.models = modelData;
  const models = (modelData && modelData.models) || [];
  const wrap = card("<div class='panel-title'>Runtime models</div>");
  if (!models.length) wrap.innerHTML += "<span class='muted'>No models reported.</span>";
  for (const m of models) {
    const loaded = m.state === "loaded";
    const node = el("div", "list-item");
    node.innerHTML =
      "<div class='row'><strong>" + esc(m.id) + "</strong>" +
      "<span class='pill'>" + esc(m.role) + "</span>" +
      "<span class='pill " + (loaded ? "ok" : "") + "'>" + esc(m.state) + "</span>" +
      (m.keep_loaded ? "<span class='pill cyan'>keep_loaded</span>" : "") +
      (m.missing_path ? "<span class='pill warn'>missing path</span>" : "") + "</div>";
    node.appendChild(modelLoadButtons(m.id));
    wrap.appendChild(node);
  }
  screenEl.appendChild(wrap);
};

screens.activity = async function () {
  screenEl.appendChild(screenHeader("Activity", "Redacted audit feed (GET /diagnostics/activity). No prompt content, file contents, tokens, or secrets."));
  const data = await api("GET", "/diagnostics/activity?limit=100");
  state.activity = (data && data.events) || [];
  const wrap = panel("Recent events", "activity-feed", "<span class='count'>" + state.activity.length + "</span>");
  screenEl.appendChild(wrap);
  const host = document.getElementById("activity-feed");
  if (!state.activity.length) {
    host.innerHTML = "<span class='muted'>No audit events yet.</span>";
    return;
  }
  const feed = el("div", "feed");
  state.activity.forEach((e) => {
    const r = D.activityRow(e);
    const bits = [];
    if (r.agent) bits.push("agent " + esc(r.agent));
    if (r.tool) bits.push("tool " + esc(r.tool));
    if (r.permission !== null) bits.push("L" + r.permission);
    if (r.outcome) bits.push(esc(r.outcome));
    if (r.ref) bits.push("#" + esc(r.ref));
    feed.appendChild(el("div", "feed-row",
      "<span class='feed-time'>" + esc(r.time) + "</span>" +
      "<span class='feed-body'><span class='feed-kind'>" + esc(r.kind) + "</span>" +
      (r.risk ? "<span class='feed-risk'>risk " + esc(r.risk) + "</span>" : "") +
      "<span class='feed-meta'>" + bits.join(" · ") + "</span></span>"));
  });
  host.appendChild(feed);
};

// ===========================================================================
//  BOOT
// ===========================================================================
function tokenErrorMessage(source) {
  if (source === "bridge-error") {
    return "Could not retrieve the API token from the desktop bridge. Re-open APRIL Desktop via `run april desktop`.";
  }
  if (source === "bridge-empty") {
    return "The desktop bridge returned an empty token. Re-open APRIL Desktop via `run april desktop`.";
  }
  return "No API token present. Launch with `run april desktop` so the token is delivered via the native bridge or the URL fragment.";
}

function showBootError(source) {
  screenEl.innerHTML = "";
  const node = el("div", "bootscreen error");
  node.innerHTML =
    "<div class='big'>APRIL DESKTOP — LOCKED</div>" +
    "<p>" + esc(tokenErrorMessage(source)) + "</p>" +
    "<p class='muted'>This UI never starts authenticated requests without a token. Everything runs on 127.0.0.1.</p>";
  screenEl.appendChild(node);
}

function showBootLoading() {
  screenEl.innerHTML = "";
  const node = el("div", "bootscreen");
  node.innerHTML = "<div class='big'>APRIL DESKTOP</div><p class='muted'>Acquiring local API token…</p>";
  screenEl.appendChild(node);
}

(async function boot() {
  bindNav();
  showBootLoading();
  let result;
  try {
    result = await window.AprilDesktopAuth.acquireToken(window);
  } catch (_err) {
    result = { token: "", source: "bridge-error" };
  }
  TOKEN = result.token;
  if (!TOKEN) {
    // Never start authenticated API clients without a token.
    document.getElementById("app").classList.remove("booting");
    showBanner(tokenErrorMessage(result.source));
    showBootError(result.source);
    return;
  }
  // Authenticated work begins only after the token has been retrieved.
  document.getElementById("app").classList.remove("booting");
  startPolling();
  navigate(DEFAULT_SCREEN);
})();
