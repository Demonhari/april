import importlib.util
import tempfile
import unittest
from pathlib import Path

from praxos.mcp_server import create_mcp_server


HAS_MCP_USE = importlib.util.find_spec("mcp_use") is not None


class MCPServerTests(unittest.TestCase):
    @unittest.skipUnless(HAS_MCP_USE, "mcp-use is not installed")
    def test_mcp_use_server_exposes_agent_tools(self):
        with tempfile.TemporaryDirectory() as td:
            server = create_mcp_server(Path(td) / "mcp.db")

        self.assertTrue(hasattr(server, "run"))
        self.assertEqual(
            {
                "check_action",
                "get_experience",
                "learn_from_feedback",
                "record_outcome",
            },
            set(server._tool_manager._tools),
        )


if __name__ == "__main__":
    unittest.main()
