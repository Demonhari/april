import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from praxos.server import create_handler


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


class ServerTests(unittest.TestCase):
    def test_tool_server_exposes_agent_loop(self):
        with tempfile.TemporaryDirectory() as td:
            handler = create_handler(Path(td) / "server.db")
            try:
                server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            except PermissionError as exc:
                raise unittest.SkipTest("socket binding is blocked in this sandbox") from exc
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                learned = post_json(
                    f"{base}/tools/learn_from_feedback",
                    {
                        "agent_id": "support-agent",
                        "task": "Reply about delivery timing",
                        "action": "Promise delivery by Friday",
                        "feedback": "Never promise delivery dates without Product approval.",
                        "source_refs": ["slack://product/date-policy"],
                    },
                )
                checked = post_json(
                    f"{base}/tools/check_action",
                    {
                        "task": "Reply about delivery timing",
                        "action": "Promise delivery by Friday",
                    },
                )
                mcp = post_json(
                    f"{base}/mcp",
                    {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertIn("result", learned)
        self.assertEqual("warn", checked["result"]["decision"])
        self.assertIn("check_action", mcp["result"]["tools"])


if __name__ == "__main__":
    unittest.main()
