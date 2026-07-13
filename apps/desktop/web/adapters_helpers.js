"use strict";

(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root) root.AprilAdapters = api;
})(typeof window !== "undefined" ? window : globalThis, function () {
  function listRequest() {
    return { method: "GET", path: "/evolution/adapters" };
  }

  function activateRequest(fields) {
    return {
      method: "POST",
      path: "/evolution/adapters/activate",
      body: {
        model_id: String(fields.model_id || "").trim(),
        adapter_path: String(fields.adapter_path || "").trim(),
        evidence_path: String(fields.evidence_path || "").trim() || null,
        verification_report_path:
          String(fields.verification_report_path || "").trim() || null,
      },
    };
  }

  function rollbackRequest(modelId, version) {
    return {
      method: "POST",
      path: "/evolution/adapters/rollback",
      body: {
        model_id: String(modelId || ""),
        version: version === undefined || version === null ? null : Number(version),
      },
    };
  }

  async function request(api, spec) {
    return api(spec.method, spec.path, spec.body);
  }

  function basename(value) {
    return String(value || "").replaceAll("\\", "/").split("/").pop() || "";
  }

  function adapterView(item) {
    item = item || {};
    const pointer = item.pointer && typeof item.pointer === "object" ? item.pointer : {};
    const versions = Array.isArray(pointer.versions) ? pointer.versions : [];
    const history = Array.isArray(item.history) ? item.history : [];
    return {
      modelId: String(item.model_id || "unknown"),
      activeVersion: pointer.active_version === undefined || pointer.active_version === null
        ? null
        : Number(pointer.active_version),
      sha256: String(pointer.sha256 || ""),
      versions: versions.map((entry) => ({
        version: Number(entry.version),
        basename: basename(entry.adapter_path),
        active: Number(entry.version) === Number(pointer.active_version),
      })),
      historyCount: history.length,
    };
  }

  return {
    activateRequest,
    adapterView,
    listRequest,
    request,
    rollbackRequest,
  };
});
