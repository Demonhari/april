"use strict";

// Behavioural tests for the DOM-free desktop dashboard helpers. Run under Node;
// the Python wrapper skips this when Node is unavailable. These lock down the
// security-relevant rules: activity redaction (defence in depth), honest
// unknown values, and content-free streaming chips.
const path = require("path");
const D = require(path.join(__dirname, "..", "..", "apps", "desktop", "web", "dashboard_helpers.js"));

let failures = 0;
function check(name, condition) {
  if (!condition) {
    failures += 1;
    console.error("FAIL: " + name);
  } else {
    console.log("ok: " + name);
  }
}

// --- agent code mapping ----------------------------------------------------
check("agent code general", D.agentCode("general_agent") === "GEN");
check("agent code coding", D.agentCode("coding_agent") === "COD");
check("agent code reading", D.agentCode("reading_agent") === "RDG");
check("agent code creative", D.agentCode("creative_agent") === "CRE");
check("agent code reasoning", D.agentCode("reasoning_agent") === "RSN");
check("agent code system", D.agentCode("system_action_agent") === "SYS");
check("agent code unknown", D.agentCode("mystery_agent") === "unknown");
check("agent code missing", D.agentCode(null) === "unknown");
check("six agents", D.AGENTS.length === 6);

// --- honest formatters -----------------------------------------------------
check("formatInt zero preserved", D.formatInt(0) === "0");
check("formatInt missing unknown", D.formatInt(undefined) === "unknown");
check("formatInt null unknown", D.formatInt(null) === "unknown");
check("formatBytes mb", D.formatBytes(5 * 1024 * 1024) === "5.0 MB");
check("formatBytes bytes", D.formatBytes(512) === "512 B");
check("formatBytes missing", D.formatBytes(undefined) === "unknown");
check("formatBytes negative", D.formatBytes(-1) === "unknown");
check("formatRate suffix", D.formatRate(12.34, " ms") === "12.3 ms");
check("formatRate missing", D.formatRate(undefined) === "unknown");

// --- telemetry: missing stays unknown, real values (incl 0) preserved ------
{
  // Minimal health (matches the fake test runtime): no telemetry available.
  const t = D.telemetryFrom({ status: "ok", runtime: { status: "ok", backend: "fake" } });
  check("telemetry tokens unknown", t.tokens_per_second === undefined);
  check("telemetry rss unknown", t.process_rss_bytes === undefined);
  check("telemetry loaded unknown", t.loaded_model_count === undefined);
}
{
  const health = {
    status: "ok",
    runtime: {
      status: "ok",
      backend: "llama_cpp",
      simulated: false,
      loaded_model_count: 0,
      active_requests: 0,
      generation_error_count: 0,
      process_rss_bytes: 1048576,
      models: [
        { state: "loaded", recent_tokens_per_second: 42.5, recent_latency_ms: 80, context_size: 4096 },
      ],
    },
  };
  const t = D.telemetryFrom(health);
  check("telemetry tokens real", t.tokens_per_second === 42.5);
  check("telemetry latency real", t.first_token_latency_ms === 80);
  check("telemetry context real", t.context_size === 4096);
  check("telemetry rss real", t.process_rss_bytes === 1048576);
  check("telemetry loaded zero preserved", t.loaded_model_count === 0);
  check("telemetry active zero preserved", t.active_requests === 0);
}

// --- backend badge is tri-state and never claims verified real model -------
check("backend simulated badge", D.backendInfo({ runtime: { simulated: true, backend: "fake" } }).badge === "SIMULATED");
check("backend simulated note not verified",
  D.backendInfo({ runtime: { simulated: true, backend: "fake" } }).note === D.NOT_VERIFIED);
check("backend real badge", D.backendInfo({ runtime: { simulated: false, backend: "llama_cpp" } }).badge === "REAL");
check("backend unknown badge", D.backendInfo({ runtime: { backend: "x" } }).badge === "unknown");
check("backend unknown when no runtime", D.backendInfo({}).backend === "unknown");

// --- subsystem status words ------------------------------------------------
{
  const health = {
    status: "ok",
    runtime: { status: "degraded", backend: "fake" },
    database: { ok: true },
    vector_index: { status: "ok" },
    voice: { status: "disabled" },
    scheduler: { enabled: true, running: false },
  };
  const s = D.subsystems(health);
  check("subsystem core api", s.core_api === "ok");
  check("subsystem runtime degraded", s.runtime === "degraded");
  check("subsystem database ok", s.database === "ok");
  check("subsystem scheduler enabled-not-running", s.scheduler === "enabled");
  check("status kind ok", D.statusKind("ok") === "ok");
  check("status kind degraded warn", D.statusKind("degraded") === "warn");
  check("status kind unavailable bad", D.statusKind("unavailable") === "bad");
}

// --- permission level ------------------------------------------------------
check("permission none", D.currentPermissionLevel(null, []) === null);
check("permission from chat", D.currentPermissionLevel(2, []) === 2);
check("permission max of chat+approval",
  D.currentPermissionLevel(1, [{ permission_level: 4 }, { permission_level: 3 }]) === 4);
check("permission ladder 0..5", D.PERMISSION_LEVELS.join(",") === "0,1,2,3,4,5");

// --- activity redaction: the security crux ---------------------------------
{
  // A worst-case audit event carrying prompt content, tool args, a path, a
  // patch, metadata, a token, and a private reason. NONE may survive.
  const hostile = {
    timestamp: "2026-06-22T00:00:00Z",
    event_type: "tool_executed",
    request_id: "req-123",
    approval_id: "appr-456",
    tool: "patch_applier",
    agent: "coding_agent",
    risk_level: "code_write",
    permission_level: 3,
    outcome: "consumed",
    arguments: { file_path: "/etc/passwd", patch: "SECRET PATCH BYTES" },
    metadata: { artifact_sha256: "deadbeef" },
    content: "USER PROMPT BODY THAT MUST NOT LEAK",
    api_token: "tok-should-never-appear",
    reason: "private reason text",
  };
  const safe = D.redactActivityEvent(hostile);
  const banned = ["arguments", "metadata", "content", "api_token", "reason", "patch", "file_path"];
  check("redact drops banned keys", banned.every((k) => !(k in safe)));
  check("redact keeps event_type", safe.event_type === "tool_executed");
  check("redact keeps tool", safe.tool === "patch_applier");
  check("redact keeps risk", safe.risk_level === "code_write");
  const blob = JSON.stringify(safe);
  const secrets = ["/etc/passwd", "SECRET PATCH BYTES", "USER PROMPT BODY", "tok-should-never-appear", "private reason"];
  check("redact serialized leaks nothing", secrets.every((s) => blob.indexOf(s) === -1));

  const row = D.activityRow(hostile);
  const rowBlob = JSON.stringify(row);
  check("activity row leaks nothing", secrets.every((s) => rowBlob.indexOf(s) === -1));
  check("activity row keeps kind", row.kind === "tool_executed");
  check("activity row keeps permission", row.permission === 3);
  check("activity row ref is approval id", row.ref === "appr-456");
}

// --- stream chips never leak message/token/decision text -------------------
{
  check("chip token is null", D.summarizeStreamEvent({ event: "token", payload: { text: "secret words" } }) === null);
  check("chip final_answer null", D.summarizeStreamEvent({ event: "final_answer", payload: { message: "secret answer" } }) === null);
  const routing = D.summarizeStreamEvent({
    event: "routing",
    payload: { agent: "coding_agent", routing_method: "llm", decision_summary: "PRIVATE SUMMARY TEXT" },
  });
  check("chip routing has code", routing.indexOf("COD") !== -1);
  check("chip routing omits decision summary", routing.indexOf("PRIVATE") === -1);
  check("chip approval", D.summarizeStreamEvent({ event: "approval_required", payload: {} }) === "approval required");
  check("chip done stop suppressed", D.summarizeStreamEvent({ event: "done", payload: { finish_reason: "stop" } }) === null);

  const patch = D.streamStateUpdate({
    event: "approval_required",
    payload: { approval: { id: "appr-9", permission_level: 4, risk_level: "code_write" }, message: "secret" },
  });
  check("stream patch approval id", patch.pendingApprovalId === "appr-9");
  check("stream patch permission", patch.lastPermissionLevel === 4);
  check("stream patch risk", patch.lastRiskLevel === "code_write");
  check("stream patch omits message", JSON.stringify(patch).indexOf("secret") === -1);

  const metaPatch = D.streamStateUpdate({
    event: "meta",
    payload: { agent: "reading_agent", model_id: "april-reading", routing_method: "fallback" },
  });
  check("stream patch active agent", metaPatch.activeAgent === "reading_agent");
  check("stream patch model id", metaPatch.lastModelId === "april-reading");
  check("stream patch route", metaPatch.lastRoute === "fallback");
}

// --- readiness/report helpers --------------------------------------------
{
  check("basename strips unix path", D.basenameOnly("/Users/hari/models/brain.gguf") === "brain.gguf");
  check("basename strips windows path", D.basenameOnly("C:\\models\\brain.gguf") === "brain.gguf");
  check("basename redacted empty", D.basenameOnly("[REDACTED]") === "");
  const notVerified = D.verificationSummary({ status: "not_verified", report: null });
  check("verification not verified title", notVerified.title === "not verified yet");
  check("verification not verified real false", notVerified.real_model_verified === false);
  check("verification not verified level none", notVerified.verification_level === "none");
  const latest = D.verificationSummary({
    status: "ok",
    message: "latest verification report",
    report: {
      generated_at: "2026-06-26T00:00:00Z",
      report_type: "multi_model",
      summary: "degraded",
      real_model_verified: false,
      verification_level: "partial",
      real_models_exercised: 1,
      real_models_passed: 1,
      skipped_count: 1,
      threshold_failure_count: 1,
      skipped: [{ name: "april-reading" }],
      threshold_failures: ["routing accuracy low"],
    },
  });
  check("verification latest report type", latest.report_type === "multi_model");
  check("verification latest level partial", latest.verification_level === "partial");
  check("verification latest core false", latest.core_or_all_verified === false);
  check("verification exercised count", latest.real_models_exercised === 1);
  check("verification passed count", latest.real_models_passed === 1);
  check("verification skipped count", latest.skipped_count === 1);
  check("verification threshold count", latest.threshold_failure_count === 1);
  const workflow = D.verificationSummary({
    status: "ok",
    report: {
      report_type: "workflow",
      summary: "pass",
      real_model_verified: false,
      checks_failed: 0,
    },
  });
  check("verification workflow report type", workflow.report_type === "workflow");
  check("verification workflow level none", workflow.verification_level === "none");
  check("verification workflow real false", workflow.real_model_verified === false);
  const core = D.verificationSummary({ status: "ok", report: { verification_level: "core" } });
  const all = D.verificationSummary({ status: "ok", report: { verification_level: "all" } });
  check("verification core true", core.core_or_all_verified === true);
  check("verification all true", all.core_or_all_verified === true);
}

// --- explicit status-screen wording (real model verified / voice / approval) ---
{
  check(
    "real model verified label none",
    D.realModelVerifiedLabel({ status: "not_verified", report: null }) ===
      "real model verified: none",
  );
  check(
    "real model verified label partial",
    D.realModelVerifiedLabel({ status: "ok", report: { verification_level: "partial" } }) ===
      "real model verified: partial",
  );
  check(
    "real model verified label all",
    D.realModelVerifiedLabel({ status: "ok", report: { verification_level: "all" } }) ===
      "real model verified: all",
  );
  // Voice is not live-verified until a report records it explicitly.
  check(
    "voice not live-verified by default",
    D.voiceLiveVerified({ status: "ok", report: { verification_level: "all" } }) === false,
  );
  check(
    "voice live warning shown when not verified",
    D.voiceLiveWarning({ status: "not_verified", report: null }).indexOf("not live-verified") !== -1,
  );
  check(
    "voice live verified clears warning",
    D.voiceLiveWarning({ status: "ok", report: { voice_live_verified: true } }) === "",
  );
  check(
    "approval disclaimer says chat yes is not approval",
    D.APPROVAL_DISCLAIMER.indexOf("not approval") !== -1 &&
      D.APPROVAL_DISCLAIMER.indexOf("run april approve") !== -1,
  );
}

// --- real-model status and voice status are derived independently -----------
{
  // A real-model report that verified at "core" and a SEPARATE voice report that
  // passed live verification. Each helper reads only its own report, so a newer
  // report of one kind never changes the other's status.
  const realModelReport = {
    status: "ok",
    report: { report_type: "multi_model", verification_level: "core", real_model_verified: true },
  };
  const voiceReport = {
    status: "ok",
    report: { report_type: "voice_live", voice_live_verified: true },
  };
  check(
    "real model label from real-model report",
    D.realModelVerifiedLabel(realModelReport) === "real model verified: core",
  );
  check("voice warning cleared from voice report", D.voiceLiveWarning(voiceReport) === "");
  // A passing voice report must NOT be read as a real-model verification.
  check(
    "voice report does not imply real model verified",
    D.realModelVerifiedLabel(voiceReport) === "real model verified: none",
  );
  // A real-model report must NOT clear the voice-not-verified warning.
  check(
    "real-model report does not clear voice warning",
    D.voiceLiveWarning(realModelReport).indexOf("not live-verified") !== -1,
  );
  const workflowReport = {
    status: "ok",
    report: { report_type: "workflow", summary: "pass", real_model_verified: false },
  };
  check(
    "workflow report does not imply real model verified",
    D.realModelVerifiedLabel(workflowReport) === "real model verified: none",
  );
}

if (failures > 0) {
  console.error(failures + " desktop dashboard checks failed");
  process.exit(1);
}
console.log("all desktop dashboard checks passed");
