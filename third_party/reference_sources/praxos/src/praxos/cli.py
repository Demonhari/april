"""Command line interface for Praxos."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from praxos.ledger import ExperienceLedger
from praxos.mcp_server import run_mcp_server
from praxos.server import run_server


def _print(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _ledger(args: argparse.Namespace) -> ExperienceLedger:
    return ExperienceLedger(Path(args.db).expanduser() if args.db else None)


def cmd_record(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    episode = ledger.record_episode(
        workspace_id=args.workspace,
        agent_id=args.agent,
        task=args.task,
        action=args.action,
        outcome=args.outcome,
        result=args.result or "",
        human_feedback=args.feedback or "",
        source_refs=args.source or [],
        account_id=args.account or "",
        customer_id=args.customer or "",
        learn=not args.no_learn,
    )
    payload = {"episode": episode.to_dict(), "stats": ledger.stats(workspace_id=args.workspace)}
    _print(payload)


def cmd_check(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    check = ledger.check_action(
        workspace_id=args.workspace,
        account_id=args.account or "",
        task=args.task,
        action=args.action,
    )
    _print(check.to_dict())


def cmd_experience(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    _print(
        ledger.get_experience(
            workspace_id=args.workspace,
            account_id=args.account or "",
            task=args.task,
            action=args.action or "",
            limit=args.limit,
        )
    )


def cmd_lesson_add(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    lesson = ledger.add_lesson(
        workspace_id=args.workspace,
        title=args.title,
        pattern=args.pattern,
        recommendation=args.recommendation,
        confidence=args.confidence,
    )
    _print(lesson.to_dict())


def cmd_policy_add(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    policy = ledger.add_policy(
        workspace_id=args.workspace,
        name=args.name,
        trigger=args.trigger,
        instruction=args.instruction,
        severity=args.severity,
    )
    _print(policy.to_dict())


def cmd_learn(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    lesson = ledger.learn_from_feedback(
        workspace_id=args.workspace,
        account_id=args.account or "",
        customer_id=args.customer or "",
        agent_id=args.agent,
        task=args.task,
        action=args.action,
        feedback=args.feedback,
        result=args.result or "",
        source_refs=args.source or [],
    )
    _print(lesson.to_dict())


def cmd_review_list(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    _print({"items": [item.to_dict() for item in ledger.list_review_items(workspace_id=args.workspace, status=args.status)]})


def cmd_review_approve(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    _print(ledger.review_item(args.id, approve=True, reviewer=args.reviewer).to_dict())


def cmd_review_reject(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    _print(ledger.review_item(args.id, approve=False, reviewer=args.reviewer).to_dict())


def cmd_business_account(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    _print(ledger.create_account(workspace_id=args.workspace, name=args.name, external_ref=args.external_ref or "").to_dict())


def cmd_business_customer(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    _print(
        ledger.create_customer(
            workspace_id=args.workspace,
            account_id=args.account,
            name=args.name,
            role=args.role or "",
            external_ref=args.external_ref or "",
        ).to_dict()
    )


def cmd_business_commitment(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    _print(
        ledger.add_commitment(
            workspace_id=args.workspace,
            account_id=args.account,
            description=args.description,
            source_uri=args.source_uri or "",
            due_at=args.due_at or "",
            status=args.status,
            confidence=args.confidence,
        ).to_dict()
    )


def cmd_business_escalation(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    _print(
        ledger.add_escalation(
            workspace_id=args.workspace,
            account_id=args.account,
            summary=args.summary,
            severity=args.severity,
            status=args.status,
            source_uri=args.source_uri or "",
        ).to_dict()
    )


def cmd_business_decision(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    _print(
        ledger.add_decision(
            workspace_id=args.workspace,
            account_id=args.account,
            decision=args.decision,
            source_uri=args.source_uri or "",
            decided_at=args.decided_at or "",
            status=args.status,
        ).to_dict()
    )


def cmd_server(args: argparse.Namespace) -> None:
    run_server(
        host=args.host,
        port=args.port,
        db_path=Path(args.db).expanduser() if args.db else None,
        workspace_id=args.workspace,
    )


def cmd_mcp(args: argparse.Namespace) -> None:
    run_mcp_server(
        db_path=Path(args.db).expanduser() if args.db else None,
        workspace_id=args.workspace,
        transport=args.transport,
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


def cmd_stats(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    _print(ledger.stats(workspace_id=args.workspace))


def cmd_demo(args: argparse.Namespace) -> None:
    ledger = _ledger(args)
    workspace = f"{args.workspace}:demo:{uuid.uuid4().hex[:8]}"

    account = ledger.create_account(
        workspace_id=workspace,
        name="Enterprise A",
        external_ref="crm://enterprise-a",
    )
    ledger.add_commitment(
        workspace_id=workspace,
        account_id=account.id,
        description="Enterprise A must not receive unapproved delivery dates for Feature X.",
        source_uri="crm://enterprise-a/commitments/feature-x",
        status="open",
        confidence=0.91,
    )
    ledger.add_escalation(
        workspace_id=workspace,
        account_id=account.id,
        summary="Feature X timing caused a prior customer escalation.",
        severity="high",
        source_uri="slack://product/escalations/2026-04-12",
    )
    ledger.add_decision(
        workspace_id=workspace,
        account_id=account.id,
        decision="Product moved Feature X to Q3; support must avoid near-term delivery promises.",
        source_uri="slack://product/escalations/2026-04-12",
        decided_at="2026-04-12",
    )
    policy = ledger.add_policy(
        workspace_id=workspace,
        name="No unverified delivery promises",
        trigger="deliver feature friday ship delivery date promise eta",
        severity="block",
        instruction="Do not promise delivery dates unless there is an approved product source.",
    )
    episode = ledger.record_episode(
        workspace_id=workspace,
        agent_id="support-agent",
        task="Reply to Enterprise A asking when Feature X ships",
        action="Tell Enterprise A that Feature X will ship by Friday",
        outcome="failure",
        result="Customer escalated after the promise was contradicted by Product.",
        human_feedback=(
            "Never promise delivery dates for Enterprise A. Product moved Feature X to Q3 "
            "in the April 12 escalation."
        ),
        source_refs=["slack://product/escalations/2026-04-12", "crm://enterprise-a"],
        account_id=account.id,
    )
    check = ledger.check_action(
        workspace_id=workspace,
        account_id=account.id,
        task="Reply to Enterprise A asking whether Feature X can be delivered this week",
        action="Say we can deliver Feature X by Friday",
    )
    if args.story:
        print("Praxos story demo")
        print()
        print("1. A support agent is about to promise Enterprise A that Feature X ships by Friday.")
        print("2. A human previously corrected this exact failure.")
        print("3. Praxos compiled that failure into a lesson, evidence receipts, and a review item.")
        print("4. Praxos checks the future action before it reaches the customer.")
        print()
        print(f"Decision: {check.decision.upper()}")
        for reason in check.reasons:
            print(f"- {reason}")
        print()
        print("This is the product: agents gain operational experience instead of repeating mistakes.")
        return
    _print(
        {
            "product": "Praxos",
            "positioning": "Experience OS for AI employees",
            "workspace": workspace,
            "account": account.to_dict(),
            "created_policy": policy.to_dict(),
            "recorded_failure": episode.to_dict(),
            "future_action_check": check.to_dict(),
            "stats": ledger.stats(workspace_id=workspace),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="praxos",
        description="Experience OS for AI employees",
    )
    parser.add_argument("--db", default=None, help="Path to Praxos SQLite database")
    parser.add_argument("--workspace", default="default", help="Workspace/account scope")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_record = sub.add_parser("record", help="Record an agent work episode")
    p_record.add_argument("--agent", required=True)
    p_record.add_argument("--task", required=True)
    p_record.add_argument("--action", required=True)
    p_record.add_argument(
        "--outcome",
        default="unknown",
        choices=["success", "failure", "blocked", "unknown"],
    )
    p_record.add_argument("--result", default="")
    p_record.add_argument("--feedback", default="")
    p_record.add_argument("--source", action="append", default=[])
    p_record.add_argument("--account", default="")
    p_record.add_argument("--customer", default="")
    p_record.add_argument("--no-learn", action="store_true")
    p_record.set_defaults(func=cmd_record)

    p_check = sub.add_parser("check", help="Check an action before an agent executes it")
    p_check.add_argument("--task", required=True)
    p_check.add_argument("--action", required=True)
    p_check.add_argument("--account", default="")
    p_check.set_defaults(func=cmd_check)

    p_exp = sub.add_parser("experience", help="Retrieve relevant experience")
    p_exp.add_argument("--task", required=True)
    p_exp.add_argument("--action", default="")
    p_exp.add_argument("--account", default="")
    p_exp.add_argument("--limit", type=int, default=5)
    p_exp.set_defaults(func=cmd_experience)

    p_learn = sub.add_parser("learn", help="Learn from human feedback")
    p_learn.add_argument("--agent", required=True)
    p_learn.add_argument("--task", required=True)
    p_learn.add_argument("--action", required=True)
    p_learn.add_argument("--feedback", required=True)
    p_learn.add_argument("--result", default="")
    p_learn.add_argument("--source", action="append", default=[])
    p_learn.add_argument("--account", default="")
    p_learn.add_argument("--customer", default="")
    p_learn.set_defaults(func=cmd_learn)

    p_lesson = sub.add_parser("lesson", help="Manage lessons")
    lesson_sub = p_lesson.add_subparsers(dest="lesson_cmd", required=True)
    p_lesson_add = lesson_sub.add_parser("add", help="Add a lesson manually")
    p_lesson_add.add_argument("--title", required=True)
    p_lesson_add.add_argument("--pattern", required=True)
    p_lesson_add.add_argument("--recommendation", required=True)
    p_lesson_add.add_argument("--confidence", type=float, default=0.7)
    p_lesson_add.set_defaults(func=cmd_lesson_add)

    p_policy = sub.add_parser("policy", help="Manage policies")
    policy_sub = p_policy.add_subparsers(dest="policy_cmd", required=True)
    p_policy_add = policy_sub.add_parser("add", help="Add an action policy")
    p_policy_add.add_argument("--name", required=True)
    p_policy_add.add_argument("--trigger", required=True)
    p_policy_add.add_argument("--instruction", required=True)
    p_policy_add.add_argument("--severity", default="warn", choices=["info", "warn", "block"])
    p_policy_add.set_defaults(func=cmd_policy_add)

    p_review = sub.add_parser("review", help="Human review queue")
    review_sub = p_review.add_subparsers(dest="review_cmd", required=True)
    p_review_list = review_sub.add_parser("list", help="List review items")
    p_review_list.add_argument("--status", default="pending", choices=["pending", "approved", "rejected"])
    p_review_list.set_defaults(func=cmd_review_list)
    p_review_approve = review_sub.add_parser("approve", help="Approve a review item")
    p_review_approve.add_argument("id")
    p_review_approve.add_argument("--reviewer", default="human")
    p_review_approve.set_defaults(func=cmd_review_approve)
    p_review_reject = review_sub.add_parser("reject", help="Reject a review item")
    p_review_reject.add_argument("id")
    p_review_reject.add_argument("--reviewer", default="human")
    p_review_reject.set_defaults(func=cmd_review_reject)

    p_business = sub.add_parser("business", help="Manage B2B SaaS business context")
    business_sub = p_business.add_subparsers(dest="business_cmd", required=True)
    p_account = business_sub.add_parser("account", help="Create an account")
    p_account.add_argument("--name", required=True)
    p_account.add_argument("--external-ref", default="")
    p_account.set_defaults(func=cmd_business_account)
    p_customer = business_sub.add_parser("customer", help="Create a customer contact")
    p_customer.add_argument("--account", required=True)
    p_customer.add_argument("--name", required=True)
    p_customer.add_argument("--role", default="")
    p_customer.add_argument("--external-ref", default="")
    p_customer.set_defaults(func=cmd_business_customer)
    p_commit = business_sub.add_parser("commitment", help="Add a customer commitment")
    p_commit.add_argument("--account", required=True)
    p_commit.add_argument("--description", required=True)
    p_commit.add_argument("--source-uri", default="")
    p_commit.add_argument("--due-at", default="")
    p_commit.add_argument("--status", default="open")
    p_commit.add_argument("--confidence", type=float, default=0.8)
    p_commit.set_defaults(func=cmd_business_commitment)
    p_escalation = business_sub.add_parser("escalation", help="Add an escalation")
    p_escalation.add_argument("--account", required=True)
    p_escalation.add_argument("--summary", required=True)
    p_escalation.add_argument("--severity", default="medium")
    p_escalation.add_argument("--status", default="open")
    p_escalation.add_argument("--source-uri", default="")
    p_escalation.set_defaults(func=cmd_business_escalation)
    p_decision = business_sub.add_parser("decision", help="Add a business decision")
    p_decision.add_argument("--account", required=True)
    p_decision.add_argument("--decision", required=True)
    p_decision.add_argument("--source-uri", default="")
    p_decision.add_argument("--decided-at", default="")
    p_decision.add_argument("--status", default="active")
    p_decision.set_defaults(func=cmd_business_decision)

    p_stats = sub.add_parser("stats", help="Show workspace stats")
    p_stats.set_defaults(func=cmd_stats)

    p_server = sub.add_parser("server", help="Start the local JSON tool server")
    p_server.add_argument("--host", default="127.0.0.1")
    p_server.add_argument("--port", type=int, default=8765)
    p_server.set_defaults(func=cmd_server)

    p_mcp = sub.add_parser("mcp", help="Start the Praxos MCP server using mcp-use")
    p_mcp.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
    p_mcp.add_argument("--host", default="127.0.0.1")
    p_mcp.add_argument("--port", type=int, default=8766)
    p_mcp.add_argument("--debug", action="store_true")
    p_mcp.set_defaults(func=cmd_mcp)

    p_demo = sub.add_parser("demo", help="Run the YC-style support-agent demo")
    p_demo.add_argument("--story", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
