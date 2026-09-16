import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from praxos import ExperienceLedger


class ExperienceLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "praxos.db"
        self.ledger = ExperienceLedger(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_db_falls_back_to_project_when_home_is_locked(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRAXOS_DATA_DIR", None)
            previous = os.getcwd()
            try:
                os.chdir(tmp)
                with patch("praxos.ledger.Path.home", return_value=Path("/proc")):
                    ledger = ExperienceLedger()
            finally:
                os.chdir(previous)

            self.assertEqual(Path(tmp) / ".praxos" / "praxos.db", ledger.db_path)
            self.assertTrue(ledger.db_path.exists())

    def test_failed_episode_creates_reusable_lesson(self):
        episode = self.ledger.record_episode(
            agent_id="support-agent",
            task="Reply to Enterprise A asking when Feature X ships",
            action="Promise Feature X will ship by Friday",
            outcome="failure",
            human_feedback="Never promise delivery dates without Product approval.",
            source_refs=["slack://product/feature-x"],
        )

        lessons = self.ledger.list_lessons()
        experience = self.ledger.get_experience(
            task="Reply to Enterprise A asking about Feature X",
            action="Promise delivery by Friday",
        )

        self.assertEqual(episode.outcome, "failure")
        self.assertEqual(1, len(lessons))
        self.assertIn(episode.id, lessons[0].evidence_episode_ids)
        self.assertTrue(lessons[0].rule)
        self.assertTrue(lessons[0].regression_case)
        self.assertTrue(experience["evidence"])

    def test_human_review_queue_can_reject_lesson(self):
        self.ledger.record_episode(
            agent_id="support-agent",
            task="Reply about roadmap",
            action="Mention private roadmap",
            outcome="failure",
            human_feedback="Do not mention private roadmap items.",
        )
        items = self.ledger.list_review_items()

        reviewed = self.ledger.review_item(items[0].id, approve=False, reviewer="pm")
        lessons = self.ledger.list_lessons()

        self.assertEqual("rejected", reviewed.status)
        self.assertEqual([], lessons)

    def test_policy_can_block_future_action(self):
        self.ledger.add_policy(
            name="No delivery promises",
            trigger="ship by friday delivery date promise",
            instruction="Do not promise dates without Product approval.",
            severity="block",
        )

        check = self.ledger.check_action(
            task="Reply to customer about Feature X",
            action="Say Feature X will ship by Friday",
        )

        self.assertEqual("block", check.decision)
        self.assertTrue(check.matched_policy_ids)

    def test_lesson_warns_on_similar_future_action(self):
        self.ledger.record_episode(
            agent_id="support-agent",
            task="Handle refund request for annual enterprise customer",
            action="Deny refund without checking contract exception",
            outcome="failure",
            human_feedback="Check enterprise contract exceptions before denying refunds.",
        )

        check = self.ledger.check_action(
            task="Handle refund request for enterprise customer",
            action="Deny refund immediately",
        )

        self.assertEqual("warn", check.decision)
        self.assertTrue(check.matched_lesson_ids)

    def test_workspaces_are_isolated(self):
        self.ledger.add_policy(
            workspace_id="acme",
            name="ACME policy",
            trigger="do not mention roadmap",
            instruction="Do not mention roadmap to ACME.",
            severity="block",
        )

        blocked = self.ledger.check_action(
            workspace_id="acme",
            task="Reply to ACME",
            action="Mention roadmap",
        )
        allowed = self.ledger.check_action(
            workspace_id="other",
            task="Reply to other customer",
            action="Mention roadmap",
        )

        self.assertEqual("block", blocked.decision)
        self.assertEqual("allow", allowed.decision)

    def test_business_context_warns_on_relevant_customer_commitment(self):
        account = self.ledger.create_account(name="Enterprise A")
        self.ledger.add_commitment(
            account_id=account.id,
            description="Do not promise Friday delivery for Feature X.",
            source_uri="crm://enterprise-a/commitments/feature-x",
        )
        self.ledger.add_decision(
            account_id=account.id,
            decision="Feature X moved to Q3.",
            source_uri="slack://product/2026-04-12",
        )
        self.ledger.add_escalation(
            account_id=account.id,
            summary="Feature X timing caused an enterprise escalation.",
            severity="high",
            source_uri="zendesk://ticket/feature-x",
        )

        check = self.ledger.check_action(
            account_id=account.id,
            task="Reply to Enterprise A about Feature X",
            action="Promise Friday delivery",
        )

        self.assertEqual("warn", check.decision)
        self.assertTrue(check.matched_business_ids)
        self.assertTrue(any("Relevant commitment" in reason for reason in check.reasons))
        self.assertTrue(any("Relevant decision" in reason for reason in check.reasons))
        self.assertTrue(any("Open escalation context" in reason for reason in check.reasons))

    def test_stats_count_core_objects(self):
        self.ledger.record_episode(
            agent_id="agent",
            task="Do task",
            action="Take action",
            outcome="success",
        )
        self.ledger.add_lesson(
            title="Manual lesson",
            pattern="task action",
            recommendation="Do it this way.",
        )
        self.ledger.add_policy(
            name="Policy",
            trigger="risky action",
            instruction="Avoid risky action.",
        )
        self.ledger.check_action(task="task", action="action")

        stats = self.ledger.stats()

        self.assertEqual(1, stats["episodes"])
        self.assertEqual(1, stats["lessons"])
        self.assertEqual(1, stats["policies"])
        self.assertEqual(1, stats["action_checks"])


if __name__ == "__main__":
    unittest.main()
