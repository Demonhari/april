import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from praxos.cli import main


class CliTests(unittest.TestCase):
    def test_demo_outputs_blocked_future_action(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "demo.db"
            out = io.StringIO()
            with redirect_stdout(out):
                main(["--db", str(db), "demo"])

            payload = json.loads(out.getvalue())

        self.assertEqual("Praxos", payload["product"])
        self.assertEqual("block", payload["future_action_check"]["decision"])
        self.assertEqual(1, payload["stats"]["episodes"])
        self.assertEqual(1, payload["stats"]["lessons"])
        self.assertEqual(1, payload["stats"]["policies"])


if __name__ == "__main__":
    unittest.main()
