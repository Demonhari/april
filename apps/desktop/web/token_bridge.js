"use strict";

// DOM-free token acquisition for the APRIL Desktop SPA. Kept separate so the
// security-sensitive bootstrap can be unit tested without a full browser.
//
// The token is never read from a synchronously-injected global, never written
// to localStorage/sessionStorage/cookies, and never logged. Native desktop
// fetches it through the async pywebview bridge; the browser path reads (and
// immediately strips) the URL fragment.
(function (root) {
  function readBrowserFragmentToken(win) {
    const hash = win.location.hash || "";
    const match = hash.match(/(?:^#|&)token=([^&]+)/);
    const token = match ? decodeURIComponent(match[1]) : "";
    // Strip the fragment so the token never lingers in the address bar/history.
    win.history.replaceState(null, "", win.location.pathname + win.location.search);
    return token;
  }

  function waitForPywebviewBridge(win, timeoutMs) {
    return new Promise(function (resolve) {
      const ready = function () {
        return Boolean(win.pywebview && win.pywebview.api);
      };
      if (ready()) {
        resolve(true);
        return;
      }
      let settled = false;
      const finish = function () {
        if (!settled) {
          settled = true;
          resolve(ready());
        }
      };
      // Handles a `pywebviewready` that fires after our scripts have executed.
      win.addEventListener("pywebviewready", finish, { once: true });
      (win.setTimeout || setTimeout)(finish, timeoutMs);
    });
  }

  async function acquireToken(win) {
    // A URL-fragment token means we are in the browser-served path.
    const fragmentToken = readBrowserFragmentToken(win);
    if (fragmentToken) return { token: fragmentToken, source: "fragment" };
    // Otherwise attempt the native bridge once it is ready.
    const ready = await waitForPywebviewBridge(win, 2000);
    if (
      !ready ||
      !win.pywebview.api ||
      typeof win.pywebview.api.get_token !== "function"
    ) {
      return { token: "", source: "none" };
    }
    try {
      const value = await win.pywebview.api.get_token();
      if (typeof value === "string" && value) return { token: value, source: "bridge" };
      return { token: "", source: "bridge-empty" };
    } catch (_err) {
      return { token: "", source: "bridge-error" };
    }
  }

  root.AprilDesktopAuth = {
    readBrowserFragmentToken: readBrowserFragmentToken,
    waitForPywebviewBridge: waitForPywebviewBridge,
    acquireToken: acquireToken,
  };

  // CommonJS export so the bootstrap can be unit tested under Node.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = root.AprilDesktopAuth;
  }
})(
  typeof window !== "undefined"
    ? window
    : typeof globalThis !== "undefined"
      ? globalThis
      : this,
);
