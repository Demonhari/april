"use strict";

// Pure, DOM-free helpers for the APRIL Desktop cockpit dashboard.
//
// Everything here is data-in / data-out so it can be unit tested under Node
// exactly like token_bridge.js. There is no DOM access, no network access, and
// no secret handling in this file. The rules these helpers encode are
// security-relevant, so they live apart from the rendering glue in app.js:
//
//   * The activity feed is allowlist-projected a SECOND time on the client
//     (defence in depth) so a server regression can never surface prompt
//     content, file contents, raw tool arguments, tokens, or secrets.
//   * Operational values are reported honestly: a missing value becomes
//     "unknown"/"not available" rather than a fabricated number. 0 is a real
//     value and is preserved.
//   * A "simulated" runtime is always labelled so it can never be mistaken for
//     a verified real model.
(function (root) {
  const UNKNOWN = "unknown";
  const UNAVAILABLE = "not available";
  const NOT_VERIFIED = "not yet verified";

  // The six known specialists rendered in the router/orbit visualisation.
  const AGENTS = [
    { id: "general_agent", code: "GEN", label: "General" },
    { id: "coding_agent", code: "COD", label: "Coding" },
    { id: "reading_agent", code: "RDG", label: "Reading" },
    { id: "creative_agent", code: "CRE", label: "Creative" },
    { id: "reasoning_agent", code: "RSN", label: "Reasoning" },
    { id: "system_action_agent", code: "SYS", label: "System" },
  ];
  const AGENT_BY_ID = AGENTS.reduce(function (acc, a) {
    acc[a.id] = a;
    return acc;
  }, {});

  function isMissing(value) {
    return value === null || value === undefined || value === "";
  }

  function agentEntry(agentId) {
    if (isMissing(agentId)) return null;
    return AGENT_BY_ID[String(agentId)] || null;
  }

  function agentCode(agentId) {
    const entry = agentEntry(agentId);
    return entry ? entry.code : UNKNOWN;
  }

  // --- honest formatters -----------------------------------------------------
  function formatInt(value) {
    if (typeof value !== "number" || !isFinite(value)) return UNKNOWN;
    return String(Math.round(value));
  }

  function formatBytes(value) {
    if (typeof value !== "number" || !isFinite(value) || value < 0) return UNKNOWN;
    if (value < 1024) return value + " B";
    const units = ["KB", "MB", "GB", "TB"];
    let n = value / 1024;
    let i = 0;
    while (n >= 1024 && i < units.length - 1) {
      n /= 1024;
      i += 1;
    }
    return n.toFixed(n >= 100 || i === 0 ? 0 : 1) + " " + units[i];
  }

  function formatRate(value, suffix) {
    if (typeof value !== "number" || !isFinite(value)) return UNKNOWN;
    return value.toFixed(value >= 100 ? 0 : 1) + (suffix || "");
  }

  function shortId(value, length) {
    if (isMissing(value)) return "";
    const text = String(value);
    const n = length || 8;
    return text.length > n ? text.slice(0, n) : text;
  }

  // --- runtime telemetry -----------------------------------------------------
  // Pull per-model telemetry out of whichever loaded model exposes it. Used by
  // both the /health.runtime payload (paths already redacted) and a richer
  // /runtime/models payload. Returns undefined when no loaded model reports it.
  function firstLoadedValue(models, key) {
    if (!Array.isArray(models)) return undefined;
    const loaded = models.filter(function (m) {
      return m && m.state === "loaded";
    });
    const pool = loaded.length ? loaded : models;
    for (const model of pool) {
      if (model && model[key] !== null && model[key] !== undefined) {
        return model[key];
      }
    }
    return undefined;
  }

  // Build the runtime telemetry view model. `health` is the /health body; an
  // optional richer `models` list (/runtime/models) is consulted for per-model
  // rates when /health did not carry them. Missing values stay undefined and
  // are rendered as "unknown" by the caller — never faked.
  function telemetryFrom(health, models) {
    const runtime = (health && health.runtime) || {};
    const runtimeModels = Array.isArray(runtime.models) ? runtime.models : null;
    const modelPool = runtimeModels || (Array.isArray(models) ? models : null);
    const pick = function (key) {
      if (runtime[key] !== null && runtime[key] !== undefined) return runtime[key];
      return firstLoadedValue(modelPool, key);
    };
    return {
      tokens_per_second: firstLoadedValue(modelPool, "recent_tokens_per_second"),
      first_token_latency_ms: firstLoadedValue(modelPool, "recent_latency_ms"),
      context_size: firstLoadedValue(modelPool, "context_size"),
      process_rss_bytes: pick("process_rss_bytes"),
      loaded_model_count: runtime.loaded_model_count,
      active_requests: runtime.active_requests,
      generation_error_count: runtime.generation_error_count,
    };
  }

  // --- backend / subsystem status -------------------------------------------
  // The simulated badge is tri-state: true (fake backend), false (real), or
  // null when /health did not report it. Real != verified; the label says so.
  function backendInfo(health) {
    const runtime = (health && health.runtime) || {};
    const simulated = typeof runtime.simulated === "boolean" ? runtime.simulated : null;
    let badge = UNKNOWN;
    if (simulated === true) badge = "SIMULATED";
    else if (simulated === false) badge = "REAL";
    return {
      backend: isMissing(runtime.backend) ? UNKNOWN : String(runtime.backend),
      simulated: simulated,
      badge: badge,
      // A real backend is still not a verified real model run.
      note: simulated === true ? NOT_VERIFIED : "",
      missing_models: Array.isArray(runtime.missing_models) ? runtime.missing_models : [],
    };
  }

  function statusWord(value) {
    if (isMissing(value)) return UNKNOWN;
    return String(value);
  }

  // Severity bucket for a status string, used to pick the indicator colour.
  function statusKind(value) {
    const word = String(value || "").toLowerCase();
    if (word === "ok" || word === "online" || word === "running" || word === "ready") return "ok";
    if (word === "degraded" || word === "starting") return "warn";
    if (
      word === "" ||
      word === UNKNOWN ||
      word === "unavailable" ||
      word === "offline" ||
      word === "error" ||
      word === "down"
    ) {
      return "bad";
    }
    return "neutral";
  }

  function subsystems(health) {
    const h = health || {};
    const runtime = h.runtime || {};
    const db = h.database || {};
    const vector = h.vector_index || {};
    const voice = h.voice || {};
    const scheduler = h.scheduler || {};
    const boolWord = function (value, okWord, badWord) {
      if (typeof value !== "boolean") return UNKNOWN;
      return value ? okWord : badWord;
    };
    return {
      core_api: statusWord(h.status),
      runtime: statusWord(runtime.status),
      backend: isMissing(runtime.backend) ? UNKNOWN : String(runtime.backend),
      database: boolWord(db.ok, "ok", "missing"),
      vector_index: statusWord(vector.status || vector.state),
      voice: statusWord(voice.status || voice.state),
      scheduler:
        typeof scheduler.enabled === "boolean"
          ? scheduler.enabled
            ? scheduler.running
              ? "running"
              : "enabled"
            : "disabled"
          : UNKNOWN,
    };
  }

  // --- permission level ------------------------------------------------------
  const PERMISSION_LEVELS = [0, 1, 2, 3, 4, 5];

  // The "current" required level is the strongest signal we have seen: the
  // highest pending approval level, otherwise the last level surfaced by chat
  // routing. Returns null when nothing elevated is in play.
  function currentPermissionLevel(lastLevel, approvals) {
    let level = null;
    const consider = function (value) {
      if (typeof value === "number" && isFinite(value)) {
        level = level === null ? value : Math.max(level, value);
      }
    };
    consider(lastLevel);
    if (Array.isArray(approvals)) {
      for (const ap of approvals) {
        if (ap) consider(ap.permission_level);
      }
    }
    return level;
  }

  // --- activity feed redaction (defence in depth) ---------------------------
  // Mirror of the server's strict allowlist. Even if the server regressed and
  // emitted prompt/argument/secret fields, the client renders ONLY these keys.
  const ACTIVITY_ALLOWED_KEYS = [
    "timestamp",
    "event_type",
    "event",
    "actor",
    "request_id",
    "audit_correlation_id",
    "approval_id",
    "reference_id",
    "reminder_id",
    "memory_id",
    "memory_type",
    "agent",
    "tool",
    "permission_level",
    "risk",
    "risk_level",
    "outcome",
    "status",
    "project_id",
    "content_length",
    "reason_length",
    "kind",
    "sink",
    "date",
  ];
  const ACTIVITY_ALLOWED_SET = ACTIVITY_ALLOWED_KEYS.reduce(function (acc, k) {
    acc[k] = true;
    return acc;
  }, {});

  function redactActivityEvent(event) {
    const safe = {};
    if (!event || typeof event !== "object") return safe;
    for (const key of Object.keys(event)) {
      if (ACTIVITY_ALLOWED_SET[key]) safe[key] = event[key];
    }
    return safe;
  }

  // Build the structural fields for one terminal row from an already-redacted
  // event. Returns only strings derived from allowlisted keys.
  function activityRow(event) {
    const safe = redactActivityEvent(event);
    const risk = safe.risk || safe.risk_level;
    return {
      kind: statusWord(safe.event_type || safe.event || "event"),
      time: isMissing(safe.timestamp) ? "" : String(safe.timestamp),
      agent: isMissing(safe.agent) ? "" : String(safe.agent),
      tool: isMissing(safe.tool) ? "" : String(safe.tool),
      risk: isMissing(risk) ? "" : String(risk),
      outcome: isMissing(safe.outcome || safe.status) ? "" : String(safe.outcome || safe.status),
      permission: typeof safe.permission_level === "number" ? safe.permission_level : null,
      ref: shortId(safe.approval_id || safe.request_id || safe.reference_id, 12),
    };
  }

  // --- chat stream events ----------------------------------------------------
  // A short, content-free chip label for a streaming event. NEVER includes
  // token text, the final message, decision summaries, or tool arguments —
  // only structural fields (agent, route method, tool name, status, risk).
  function summarizeStreamEvent(evt) {
    if (!evt || typeof evt !== "object") return null;
    const event = evt.event;
    const payload = evt.payload || {};
    if (event === "meta" || event === "routing") {
      const code = agentCode(payload.agent);
      const method = isMissing(payload.routing_method) ? "" : " · " + payload.routing_method;
      return "route · " + code + method;
    }
    if (event === "agent_iteration") {
      return agentCode(payload.agent) + " · " + statusWord(payload.status);
    }
    if (event === "tool_request") {
      return "tool · " + statusWord(payload.tool);
    }
    if (event === "approval_required") {
      return "approval required";
    }
    if (event === "error") {
      return "error · " + statusWord(payload.code || payload.status || "generation failed");
    }
    if (event === "done") {
      const reason = statusWord(payload.finish_reason);
      return reason === "stop" ? null : "done · " + reason;
    }
    // token / usage / final_answer carry user-visible text handled elsewhere.
    return null;
  }

  // Derive a dashboard state patch from a streaming event. Pure: returns only
  // the keys that change, leaving content/text untouched.
  function streamStateUpdate(evt) {
    if (!evt || typeof evt !== "object") return {};
    const event = evt.event;
    const payload = evt.payload || {};
    const patch = {};
    if (event === "meta" || event === "routing") {
      if (!isMissing(payload.agent)) patch.activeAgent = String(payload.agent);
      if (!isMissing(payload.model_id)) patch.lastModelId = String(payload.model_id);
      if (!isMissing(payload.routing_method)) patch.lastRoute = String(payload.routing_method);
    } else if (event === "agent_iteration") {
      if (!isMissing(payload.agent)) patch.activeAgent = String(payload.agent);
      if (!isMissing(payload.status)) patch.lastDecision = String(payload.status);
    } else if (event === "approval_required") {
      const ap = payload.approval || {};
      if (!isMissing(ap.id)) patch.pendingApprovalId = String(ap.id);
      if (typeof ap.permission_level === "number") patch.lastPermissionLevel = ap.permission_level;
      if (!isMissing(ap.risk_level)) patch.lastRiskLevel = String(ap.risk_level);
    } else if (event === "done") {
      if (!isMissing(payload.finish_reason)) patch.lastDecision = String(payload.finish_reason);
    }
    return patch;
  }

  function basenameOnly(value) {
    if (isMissing(value)) return "";
    const text = String(value);
    if (text === "[REDACTED]") return "";
    const normalized = text.replace(/\\/g, "/");
    const parts = normalized.split("/").filter(Boolean);
    return parts.length ? parts[parts.length - 1] : normalized;
  }

  function verificationSummary(latest) {
    if (!latest || latest.status === "not_verified" || !latest.report) {
      return {
        status: "not_verified",
        title: "not verified yet",
        summary: "degraded",
        real_model_verified: false,
        report_type: "none",
        verification_level: "none",
        core_or_all_verified: false,
        skipped_count: 0,
        threshold_failure_count: 0,
      };
    }
    const report = latest.report || {};
    const level = ["none", "partial", "core", "all"].indexOf(report.verification_level) === -1
      ? "none"
      : report.verification_level;
    return {
      status: statusWord(latest.status),
      title: statusWord(latest.message || "latest verification report"),
      generated_at: statusWord(report.generated_at),
      report_type: statusWord(report.report_type),
      summary: statusWord(report.summary),
      real_model_verified: report.real_model_verified === true,
      verification_level: level,
      core_or_all_verified: level === "core" || level === "all",
      real_models_exercised: typeof report.real_models_exercised === "number" ? report.real_models_exercised : 0,
      real_models_passed: typeof report.real_models_passed === "number" ? report.real_models_passed : 0,
      skipped_count: typeof report.skipped_count === "number"
        ? report.skipped_count
        : Array.isArray(report.skipped) ? report.skipped.length : 0,
      threshold_failure_count: typeof report.threshold_failure_count === "number"
        ? report.threshold_failure_count
        : Array.isArray(report.threshold_failures) ? report.threshold_failures.length : 0,
    };
  }

  root.AprilDashboard = {
    UNKNOWN: UNKNOWN,
    UNAVAILABLE: UNAVAILABLE,
    NOT_VERIFIED: NOT_VERIFIED,
    AGENTS: AGENTS,
    PERMISSION_LEVELS: PERMISSION_LEVELS,
    ACTIVITY_ALLOWED_KEYS: ACTIVITY_ALLOWED_KEYS,
    isMissing: isMissing,
    agentEntry: agentEntry,
    agentCode: agentCode,
    formatInt: formatInt,
    formatBytes: formatBytes,
    formatRate: formatRate,
    shortId: shortId,
    telemetryFrom: telemetryFrom,
    backendInfo: backendInfo,
    statusWord: statusWord,
    statusKind: statusKind,
    subsystems: subsystems,
    currentPermissionLevel: currentPermissionLevel,
    redactActivityEvent: redactActivityEvent,
    activityRow: activityRow,
    summarizeStreamEvent: summarizeStreamEvent,
    streamStateUpdate: streamStateUpdate,
    basenameOnly: basenameOnly,
    verificationSummary: verificationSummary,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = root.AprilDashboard;
  }
})(
  typeof window !== "undefined"
    ? window
    : typeof globalThis !== "undefined"
      ? globalThis
      : this,
);
