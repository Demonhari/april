"use strict";

const path = require("path");
const A = require(path.join(
  __dirname,
  "..",
  "..",
  "apps",
  "desktop",
  "web",
  "adapters_helpers.js",
));

let failures = 0;
function check(name, condition) {
  if (!condition) {
    failures += 1;
    console.error("FAIL: " + name);
  } else {
    console.log("ok: " + name);
  }
}

async function main() {
  const view = A.adapterView({
    model_id: "april-brain",
    pointer: {
      active_version: 2,
      sha256: "abc123",
      versions: [
        { version: 1, adapter_path: "/private/models/brain-v1.gguf" },
        { version: 2, adapter_path: "/private/models/brain-v2.gguf" },
      ],
    },
    history: [{ id: "one" }, { id: "two" }],
  });
  check("renders model id", view.modelId === "april-brain");
  check("renders active version", view.activeVersion === 2);
  check("renders version rows", view.versions.length === 2);
  check("marks active version", view.versions[1].active === true);
  check("renders basename only", view.versions[0].basename === "brain-v1.gguf");
  check("renders history count", view.historyCount === 2);

  const calls = [];
  async function stubFetch(method, requestPath, body) {
    calls.push({ method, path: requestPath, body });
    return { ok: true };
  }

  await A.request(stubFetch, A.listRequest());
  await A.request(stubFetch, A.activateRequest({
    model_id: "april-brain",
    adapter_path: "/models/brain.gguf",
    evidence_path: "/evidence/ppl.json",
    verification_report_path: "/reports/real.json",
  }));
  await A.request(stubFetch, A.rollbackRequest("april-brain", null));

  check(
    "list request wiring",
    calls[0].method === "GET" && calls[0].path === "/evolution/adapters",
  );
  check(
    "activate request wiring",
    calls[1].method === "POST" &&
      calls[1].path === "/evolution/adapters/activate" &&
      calls[1].body.model_id === "april-brain" &&
      calls[1].body.evidence_path === "/evidence/ppl.json" &&
      calls[1].body.verification_report_path === "/reports/real.json",
  );
  check(
    "rollback request wiring",
    calls[2].method === "POST" &&
      calls[2].path === "/evolution/adapters/rollback" &&
      calls[2].body.model_id === "april-brain" &&
      calls[2].body.version === null,
  );

  if (failures > 0) {
    console.error(failures + " desktop adapter checks failed");
    process.exit(1);
  }
  console.log("all desktop adapter checks passed");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
