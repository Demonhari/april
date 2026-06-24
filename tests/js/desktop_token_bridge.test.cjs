"use strict";

// Behavioural tests for the DOM-free desktop token bootstrap. Run under Node;
// the Python wrapper skips this when Node is unavailable.
const path = require("path");
const auth = require(path.join(__dirname, "..", "..", "apps", "desktop", "web", "token_bridge.js"));

let failures = 0;
function check(name, condition) {
  if (!condition) {
    failures += 1;
    console.error("FAIL: " + name);
  } else {
    console.log("ok: " + name);
  }
}

function makeWindow(opts) {
  opts = opts || {};
  const listeners = {};
  const timeouts = [];
  const win = {
    location: { hash: opts.hash || "", pathname: "/desktop", search: "" },
    history: {
      replaceCalls: 0,
      replaceState() {
        this.replaceCalls += 1;
      },
    },
    pywebview: opts.pywebview,
    addEventListener(type, cb) {
      (listeners[type] = listeners[type] || []).push(cb);
    },
    dispatch(type) {
      (listeners[type] || []).forEach((cb) => cb());
    },
    setTimeout(cb) {
      timeouts.push(cb);
      return timeouts.length;
    },
    flushTimers() {
      timeouts.splice(0).forEach((cb) => cb());
    },
  };
  // Any access to a persistent store is a security regression.
  Object.defineProperty(win, "localStorage", {
    get() {
      throw new Error("localStorage accessed");
    },
  });
  Object.defineProperty(win, "sessionStorage", {
    get() {
      throw new Error("sessionStorage accessed");
    },
  });
  win.document = {
    get cookie() {
      throw new Error("cookie accessed");
    },
    set cookie(_v) {
      throw new Error("cookie written");
    },
  };
  return win;
}

async function main() {
  // Browser mode: token from URL fragment, then fragment stripped.
  {
    const win = makeWindow({ hash: "#token=abc123" });
    const result = await auth.acquireToken(win);
    check("browser fragment token", result.source === "fragment" && result.token === "abc123");
    check("browser fragment stripped", win.history.replaceCalls === 1);
  }

  // Native: bridge becomes ready only after a delayed pywebviewready event.
  {
    const win = makeWindow({});
    const pending = auth.acquireToken(win);
    win.pywebview = { api: { get_token: async () => "tok-delayed" } };
    win.dispatch("pywebviewready");
    const result = await pending;
    check("delayed pywebviewready", result.source === "bridge" && result.token === "tok-delayed");
  }

  // Native: bridge ready immediately.
  {
    const win = makeWindow({ pywebview: { api: { get_token: async () => "tok-now" } } });
    const result = await auth.acquireToken(win);
    check("immediate bridge token", result.source === "bridge" && result.token === "tok-now");
  }

  // Bridge failure: get_token throws.
  {
    const win = makeWindow({
      pywebview: {
        api: {
          get_token: async () => {
            throw new Error("boom");
          },
        },
      },
    });
    const result = await auth.acquireToken(win);
    check("bridge failure", result.source === "bridge-error" && result.token === "");
  }

  // Empty token from the bridge.
  {
    const win = makeWindow({ pywebview: { api: { get_token: async () => "" } } });
    const result = await auth.acquireToken(win);
    check("empty bridge token", result.source === "bridge-empty" && result.token === "");
  }

  // No bridge and no fragment: times out to no token.
  {
    const win = makeWindow({});
    const pending = auth.acquireToken(win);
    win.flushTimers();
    const result = await pending;
    check("no token without bridge or fragment", result.source === "none" && result.token === "");
  }

  if (failures > 0) {
    console.error(failures + " desktop token bridge checks failed");
    process.exit(1);
  }
  console.log("all desktop token bridge checks passed");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
