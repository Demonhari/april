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
// (window.AprilDesktopAuth) so it can be unit tested in isolation.
let TOKEN = "";

const BASE = window.location.origin;
const CONVERSATION_ID = (crypto.randomUUID && crypto.randomUUID()) ||
  String(Date.now()) + "-" + Math.random().toString(16).slice(2);

let selectedProject = null; // {id, name, path} or null

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

function showBanner(message) {
  bannerText.textContent = message;
  bannerEl.classList.add("show");
}
function clearBanner() {
  bannerEl.classList.remove("show");
  bannerText.textContent = "";
}
$("#banner-dismiss").addEventListener("click", clearBanner);

function authHeaders(extra) {
  return Object.assign({ Authorization: "Bearer " + TOKEN }, extra || {});
}

// --- API helper ------------------------------------------------------------
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

// --- Connection indicator --------------------------------------------------
async function refreshConnection() {
  try {
    const data = await fetch(BASE + "/health").then((r) => r.json());
    const dot = data.status === "ok" ? "●" : "◐";
    $("#conn").textContent = dot + " api " + data.status + " · 127.0.0.1";
  } catch (_) {
    $("#conn").textContent = "○ api unreachable";
  }
}

// --- Screen registry -------------------------------------------------------
const screens = {};
let activeScreen = "chat";

function setActive(name) {
  activeScreen = name;
  document.querySelectorAll("nav .navitem").forEach((b) => {
    b.classList.toggle("active", b.dataset.screen === name);
  });
}

async function navigate(name) {
  setActive(name);
  screenEl.innerHTML = "";
  try {
    await screens[name]();
  } catch (err) {
    // Errors already surfaced via banner; keep the UI alive.
    if (!String(err.message || "").startsWith("auth-") && !String(err.message || "").startsWith("http-")) {
      console.error(err);
    }
  }
}

document.querySelectorAll("nav .navitem").forEach((b) => {
  b.addEventListener("click", () => navigate(b.dataset.screen));
});

function header(title, sub) {
  const h = document.createElement("div");
  h.innerHTML = "<h1>" + esc(title) + "</h1><p class='sub'>" + esc(sub || "") + "</p>";
  return h;
}
function card(html) {
  const c = document.createElement("div");
  c.className = "card";
  c.innerHTML = html;
  return c;
}

// --- Chat ------------------------------------------------------------------
let chatBusy = false;

screens.chat = async function () {
  screenEl.appendChild(header("Chat", "Streams from POST /chat/stream over loopback. One conversation per session."));

  const projLabel = selectedProject
    ? "Project scope: " + esc(selectedProject.name)
    : "No project selected (general chat). Pick one under Projects.";
  const bar = card(
    "<div class='row'><span class='pill'>" + projLabel + "</span>" +
    "<span class='grow'></span><span class='kv'>conversation " + esc(CONVERSATION_ID.slice(0, 8)) + "</span></div>"
  );
  screenEl.appendChild(bar);

  const log = document.createElement("div");
  log.id = "chat-log";
  screenEl.appendChild(log);

  const compose = card(
    "<textarea id='chat-input' placeholder='Ask APRIL… (Enter to send, Shift+Enter for newline)'></textarea>" +
    "<div class='spacer'></div><div class='row'><button class='btn' id='chat-send'>Send</button>" +
    "<small class='hint'>Approvals are never granted from chat. Use the Approvals screen.</small></div>"
  );
  screenEl.appendChild(compose);

  const input = $("#chat-input");
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });
  $("#chat-send").addEventListener("click", sendChat);
  input.focus();
};

function chatLine(kind, html) {
  const log = $("#chat-log");
  if (!log) return null;
  const el = document.createElement("div");
  el.className = "msg " + kind;
  el.innerHTML = html;
  log.appendChild(el);
  el.scrollIntoView({ block: "end" });
  return el;
}

async function sendChat() {
  if (chatBusy) return;
  const input = $("#chat-input");
  const message = (input.value || "").trim();
  if (!message) return;
  input.value = "";
  chatBusy = true;
  $("#chat-send").disabled = true;
  chatLine("user", esc(message));
  const assistant = chatLine("assistant", "<span class='muted'>…</span>");
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
      assistant.classList.replace("assistant", "errline");
      assistant.textContent = "Authentication failed.";
      return;
    }
    if (!res.ok || !res.body) {
      assistant.classList.replace("assistant", "errline");
      assistant.textContent = "Stream failed (" + res.status + ").";
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
        assistantText = handleChatEvent(evt, assistant, assistantText);
      }
    }
  } catch (err) {
    showBanner("Chat stream network error. Is the local API running?");
    assistant.classList.replace("assistant", "errline");
    assistant.textContent = "Network error during stream.";
  } finally {
    chatBusy = false;
    const sendBtn = $("#chat-send");
    if (sendBtn) sendBtn.disabled = false;
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

function handleChatEvent(evt, assistant, assistantText) {
  const event = evt.event;
  const payload = evt.payload || {};
  if (event === "token") {
    assistantText += payload.text || "";
    assistant.textContent = assistantText;
    assistant.scrollIntoView({ block: "end" });
  } else if (event === "approval_required") {
    const ap = payload.approval || {};
    const id = ap.id || (payload.message ? "" : "");
    chatLine(
      "approval",
      "<strong>Approval required.</strong> APRIL paused and will not act without explicit, " +
      "exact-ID approval.<div class='spacer'></div><span class='kv'>approval_id: " + esc(id) + "</span>" +
      "<div class='spacer'></div><button class='btn' id='goto-approvals'>Review in Approvals</button>"
    );
    const btn = $("#goto-approvals");
    if (btn) btn.addEventListener("click", () => navigate("approvals"));
  } else if (event === "error") {
    chatLine("errline", "Error: " + esc(payload.message || payload.code || "generation failed"));
  } else if (event === "usage" || event === "done" || event === "meta" ||
             event === "decision" || event === "final_answer") {
    // surfaced subtly; tokens already carry the user-visible text
    if (event === "done" && payload.finish_reason && payload.finish_reason !== "stop") {
      chatLine("status", "done: " + esc(payload.finish_reason));
    }
  } else if (event === "agent_iteration") {
    chatLine("status", esc(payload.agent || "agent") + " · " + esc(payload.status || ""));
  } else if (event === "tool_request") {
    chatLine("status", "tool requested: " + esc(payload.tool || "(tool)"));
  }
  return assistantText;
}

// --- Projects --------------------------------------------------------------
screens.projects = async function () {
  screenEl.appendChild(header("Projects", "Selecting a project scopes Chat and coding work. Paths must resolve inside allowed roots."));
  const data = await api("GET", "/projects");
  const projects = (data && data.projects) || [];

  const add = card(
    "<div class='row'><input class='grow' id='proj-path' placeholder='/absolute/path/to/repo' />" +
    "<input id='proj-name' placeholder='name (optional)' />" +
    "<button class='btn' id='proj-add'>Add project</button></div>"
  );
  screenEl.appendChild(add);
  $("#proj-add").addEventListener("click", async () => {
    const path = $("#proj-path").value.trim();
    if (!path) return;
    const name = $("#proj-name").value.trim();
    await api("POST", "/projects", { path, name: name || null });
    navigate("projects");
  });

  const list = card("<strong>Registered projects</strong>");
  if (!projects.length) {
    list.innerHTML += "<div class='spacer'></div><span class='muted'>No projects yet.</span>";
  }
  for (const p of projects) {
    const item = document.createElement("div");
    item.className = "list-item";
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

function showBannerInfo(message) {
  // reuse the banner element for transient info without the error styling intent
  bannerText.textContent = message;
  bannerEl.classList.add("show");
  setTimeout(clearBanner, 2500);
}

// --- Approvals -------------------------------------------------------------
screens.approvals = async function () {
  screenEl.appendChild(header("Approvals", "Exact-ID, one-time approvals. A chat 'yes' is never approval."));
  const data = await api("GET", "/approvals");
  const approvals = (data && data.approvals) || [];
  const wrap = card("<strong>Pending approvals</strong>");
  if (!approvals.length) {
    wrap.innerHTML += "<div class='spacer'></div><span class='muted'>Nothing awaiting approval.</span>";
  }
  for (const ap of approvals) {
    const item = document.createElement("div");
    item.className = "list-item";
    const args = ap.args || {};
    const meta = ap.metadata || {};
    const digest = ap.canonical_hash || meta.artifact_sha256 || meta.digest || "";
    const paths = collectPaths(args).concat(collectPaths(meta));
    item.innerHTML =
      "<div class='row'><span class='pill warn'>" + esc(ap.tool) + "</span>" +
      "<span class='pill'>risk " + esc(ap.risk_level) + "</span>" +
      "<span class='pill'>level " + esc(ap.permission_level) + "</span>" +
      "<span class='grow'></span><span class='kv'>expires " + esc(ap.expires_at) + "</span></div>" +
      "<div class='kv'>approval_id " + esc(ap.id) + "</div>" +
      (digest ? "<div class='kv'>digest " + esc(digest) + "</div>" : "") +
      (paths.length ? "<div class='kv'>paths: " + esc(paths.join(", ")) + "</div>" : "") +
      "<pre>" + esc(JSON.stringify(args, null, 2)) + "</pre>" +
      "<div class='row'><button class='btn' data-act='approve'>Approve this exact ID</button>" +
      "<button class='btn danger' data-act='deny'>Deny</button></div>";
    item.querySelector("[data-act='approve']").addEventListener("click", async () => {
      await api("POST", "/tools/approve", { approval_id: ap.id });
      showBannerInfo("Approved " + ap.id + ".");
      navigate("approvals");
    });
    item.querySelector("[data-act='deny']").addEventListener("click", async () => {
      await api("POST", "/tools/deny", { approval_id: ap.id });
      showBannerInfo("Denied " + ap.id + ".");
      navigate("approvals");
    });
    wrap.appendChild(item);
  }
  screenEl.appendChild(wrap);
};

function collectPaths(obj) {
  const out = [];
  if (!obj || typeof obj !== "object") return out;
  for (const [k, v] of Object.entries(obj)) {
    if (typeof v === "string" && /(^|_)path$|repo_path|folder/.test(k)) out.push(v);
  }
  return out;
}

// --- Memory ----------------------------------------------------------------
screens.memory = async function () {
  screenEl.appendChild(header("Memory", "Explicit durable local memory. Nothing is auto-created; you see exactly what is stored."));

  const search = card(
    "<div class='row'><input class='grow' id='mem-q' placeholder='search memories' value='*' />" +
    "<button class='btn' id='mem-search'>Search</button>" +
    "<button class='btn secondary' id='mem-export'>Export all</button></div>"
  );
  screenEl.appendChild(search);

  const create = card(
    "<strong>Store a memory</strong><div class='spacer'></div>" +
    "<textarea id='mem-content' placeholder='content to remember'></textarea><div class='spacer'></div>" +
    "<div class='row'><select id='mem-type'><option value='fact'>fact</option>" +
    "<option value='preference'>preference</option></select>" +
    "<input class='grow' id='mem-reason' placeholder='reason (why store this)' />" +
    "<button class='btn' id='mem-add'>Store</button></div>"
  );
  screenEl.appendChild(create);

  const results = document.createElement("div");
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
    results.appendChild(card("<strong>Export</strong><pre>" + esc(JSON.stringify(data.export, null, 2)) + "</pre>"));
  });
  $("#mem-add").addEventListener("click", async () => {
    const content = $("#mem-content").value.trim();
    const reason = $("#mem-reason").value.trim();
    if (!content || !reason) {
      showBanner("Both content and reason are required to store a memory.");
      return;
    }
    await api("POST", "/memory", {
      content,
      reason,
      memory_type: $("#mem-type").value,
      project_id: selectedProject ? selectedProject.id : null,
    });
    $("#mem-content").value = "";
    $("#mem-reason").value = "";
    showBannerInfo("Memory stored.");
    runSearch();
  });

  function renderMemoryList(items) {
    results.innerHTML = "";
    const wrap = card("<strong>" + items.length + " memor" + (items.length === 1 ? "y" : "ies") + "</strong>");
    for (const m of items) {
      const el = document.createElement("div");
      el.className = "list-item";
      el.innerHTML =
        "<div class='row'><span class='pill'>" + esc(m.kind) + "</span>" +
        (m.project_id ? "<span class='pill'>project " + esc(String(m.project_id).slice(0, 8)) + "</span>" : "") +
        "<span class='grow'></span><button class='btn danger' data-act='del'>Delete</button></div>" +
        "<div>" + esc(m.content) + "</div>" +
        "<div class='kv'>reason: " + esc(m.reason) + "</div><div class='kv'>id " + esc(m.id) + "</div>";
      el.querySelector("[data-act='del']").addEventListener("click", async () => {
        await api("DELETE", "/memory/" + encodeURIComponent(m.id));
        showBannerInfo("Deleted memory.");
        runSearch();
      });
      wrap.appendChild(el);
    }
    results.appendChild(wrap);
  }

  await runSearch();
};

// --- Reminders & Tasks -----------------------------------------------------
screens.reminders = async function () {
  screenEl.appendChild(header("Reminders & Tasks", "Local reminders, inspectable task plans, and today's briefing."));

  // Briefing
  const briefingCard = card("<strong>Today's briefing</strong><div class='spacer'></div><span class='muted'>loading…</span>");
  screenEl.appendChild(briefingCard);
  try {
    const b = await api("GET", "/scheduler/briefing/preview");
    briefingCard.innerHTML =
      "<strong>Today's briefing</strong><div class='spacer'></div>" +
      "<div><strong>" + esc(b.title || "Briefing") + "</strong></div>" +
      "<pre>" + esc(b.body || "") + "</pre>";
  } catch (_) {
    briefingCard.innerHTML = "<strong>Today's briefing</strong><div class='spacer'></div><span class='muted'>unavailable</span>";
  }

  // Reminder create
  const add = card(
    "<strong>New reminder</strong><div class='spacer'></div><div class='row'>" +
    "<input class='grow' id='rem-content' placeholder='reminder content' />" +
    "<input id='rem-due' placeholder='due_at ISO (optional)' />" +
    "<button class='btn' id='rem-add'>Add</button></div>"
  );
  screenEl.appendChild(add);
  $("#rem-add").addEventListener("click", async () => {
    const content = $("#rem-content").value.trim();
    if (!content) return;
    const due = $("#rem-due").value.trim();
    await api("POST", "/reminders", { content, due_at: due || null });
    navigate("reminders");
  });

  const remData = await api("GET", "/reminders");
  const reminders = (remData && remData.reminders) || [];
  const remWrap = card("<strong>Reminders</strong>");
  if (!reminders.length) remWrap.innerHTML += "<div class='spacer'></div><span class='muted'>None.</span>";
  for (const r of reminders) {
    const el = document.createElement("div");
    el.className = "list-item";
    el.innerHTML =
      "<div class='row'><span class='grow'>" + esc(r.content) + "</span>" +
      "<button class='btn danger' data-act='del'>Delete</button></div>" +
      "<div class='kv'>" + (r.due_at ? "due " + esc(r.due_at) : "no due date") +
      (r.fired_at ? " · fired " + esc(r.fired_at) : "") + " · id " + esc(r.id) + "</div>";
    el.querySelector("[data-act='del']").addEventListener("click", async () => {
      await api("DELETE", "/reminders/" + encodeURIComponent(r.id));
      navigate("reminders");
    });
    remWrap.appendChild(el);
  }
  screenEl.appendChild(remWrap);

  const taskData = await api("GET", "/tasks");
  const tasks = (taskData && taskData.tasks) || [];
  const taskWrap = card("<strong>Tasks</strong>");
  if (!tasks.length) taskWrap.innerHTML += "<div class='spacer'></div><span class='muted'>No task plans.</span>";
  for (const t of tasks) {
    const el = document.createElement("div");
    el.className = "list-item";
    el.innerHTML = "<pre>" + esc(JSON.stringify(t, null, 2)) + "</pre>";
    taskWrap.appendChild(el);
  }
  screenEl.appendChild(taskWrap);
};

// --- Status & Models -------------------------------------------------------
screens.status = async function () {
  screenEl.appendChild(header("Status & Models", "Local health, diagnostics, and runtime model control."));

  const health = await api("GET", "/health").catch(() => null);
  if (health) {
    const rt = health.runtime || {};
    const simulated = rt.simulated === true;
    const badge = simulated
      ? "<span class='pill'>SIMULATED runtime (fake backend) — not real-model verified</span>"
      : "<span class='pill ok'>real backend</span>";
    let note = "<strong>Runtime mode</strong> " + badge;
    if (Array.isArray(rt.missing_models) && rt.missing_models.length) {
      note += "<div class='spacer'></div><span class='muted'>Configured but missing model files: " +
        esc(rt.missing_models.join(", ")) + "</span>";
    }
    screenEl.appendChild(card(note));
    screenEl.appendChild(card("<strong>Health (redacted /health)</strong><pre>" + esc(JSON.stringify(health, null, 2)) + "</pre>"));
  }

  try {
    const diag = await api("GET", "/diagnostics");
    screenEl.appendChild(card("<strong>Diagnostics</strong><pre>" + esc(JSON.stringify(diag, null, 2)) + "</pre>"));
  } catch (_) { /* surfaced via banner */ }

  const modelData = await api("GET", "/runtime/models").catch(() => null);
  const models = (modelData && modelData.models) || [];
  const wrap = card("<strong>Runtime models</strong>");
  if (!models.length) wrap.innerHTML += "<div class='spacer'></div><span class='muted'>No models reported.</span>";
  for (const m of models) {
    const el = document.createElement("div");
    el.className = "list-item";
    const loaded = m.state === "loaded";
    el.innerHTML =
      "<div class='row'><strong>" + esc(m.id) + "</strong>" +
      "<span class='pill'>" + esc(m.role) + "</span>" +
      "<span class='pill " + (loaded ? "ok" : "") + "'>" + esc(m.state) + "</span>" +
      (m.keep_loaded ? "<span class='pill'>keep_loaded</span>" : "") +
      "<span class='grow'></span>" +
      "<button class='btn secondary' data-act='load'>Load</button>" +
      "<button class='btn secondary' data-act='unload'>Unload</button></div>";
    el.querySelector("[data-act='load']").addEventListener("click", async () => {
      await api("POST", "/runtime/models/load", { model_id: m.id });
      navigate("status");
    });
    el.querySelector("[data-act='unload']").addEventListener("click", async () => {
      await api("POST", "/runtime/models/unload", { model_id: m.id });
      navigate("status");
    });
    wrap.appendChild(el);
  }
  screenEl.appendChild(wrap);
};

// --- Activity / Logs -------------------------------------------------------
screens.activity = async function () {
  screenEl.appendChild(header("Activity", "Redacted audit feed (GET /diagnostics/activity). No prompt content, file contents, tokens, or secrets."));
  const data = await api("GET", "/diagnostics/activity?limit=100");
  const events = (data && data.events) || [];
  const wrap = card("<strong>" + events.length + " recent events</strong>");
  if (!events.length) wrap.innerHTML += "<div class='spacer'></div><span class='muted'>No audit events yet.</span>";
  for (const e of events) {
    const el = document.createElement("div");
    el.className = "list-item";
    const kind = e.event_type || e.event || "event";
    const risk = e.risk || e.risk_level;
    el.innerHTML =
      "<div class='row'><span class='pill'>" + esc(kind) + "</span>" +
      (risk ? "<span class='pill warn'>risk " + esc(risk) + "</span>" : "") +
      (e.outcome ? "<span class='pill'>" + esc(e.outcome) + "</span>" : "") +
      "<span class='grow'></span><span class='kv'>" + esc(e.timestamp || "") + "</span></div>" +
      "<div class='kv'>" +
      (e.tool ? "tool " + esc(e.tool) + " · " : "") +
      (e.agent ? "agent " + esc(e.agent) + " · " : "") +
      (e.approval_id ? "approval " + esc(e.approval_id) + " · " : "") +
      (e.request_id ? "req " + esc(e.request_id) : "") +
      "</div>";
    wrap.appendChild(el);
  }
  screenEl.appendChild(wrap);
};

// --- Boot ------------------------------------------------------------------
function tokenErrorMessage(source) {
  if (source === "bridge-error") {
    return "Could not retrieve the API token from the desktop bridge. Re-open APRIL Desktop via `run april desktop`.";
  }
  if (source === "bridge-empty") {
    return "The desktop bridge returned an empty token. Re-open APRIL Desktop via `run april desktop`.";
  }
  return "No API token present. Launch with `run april desktop` so the token is delivered via the native bridge or the URL fragment.";
}

(async function boot() {
  let result;
  try {
    result = await window.AprilDesktopAuth.acquireToken(window);
  } catch (_err) {
    result = { token: "", source: "bridge-error" };
  }
  TOKEN = result.token;
  if (!TOKEN) {
    // Never start authenticated API clients without a token.
    showBanner(tokenErrorMessage(result.source));
    return;
  }
  // Authenticated work begins only after the token has been retrieved.
  refreshConnection();
  setInterval(refreshConnection, 10000);
})();
navigate("chat");
