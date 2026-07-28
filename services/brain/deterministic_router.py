from __future__ import annotations

import re
from dataclasses import dataclass

from agents.schemas import AgentName
from services.brain.schemas import BrainDecision, PlannedToolCall
from services.permissions.schemas import RiskLevel

_APPROVAL_ID = r"[A-Za-z0-9][A-Za-z0-9_-]{7,127}"
_PATH = r"(?![/~])(?!.*(?:^|/)\.\.(?:/|$))[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*"


@dataclass(frozen=True, slots=True)
class DeterministicMatch:
    decision: BrainDecision
    matched_rule: str


class DeterministicRouter:
    """Conservative anchored routes that are safe to select before a model."""

    def route(self, message: str) -> DeterministicMatch | None:
        text = " ".join(message.strip().split())
        lowered = text.lower()

        safety = self._safety_route(lowered)
        if safety is not None:
            return safety

        approval = re.fullmatch(
            rf"(?:approve|approval)\s+(?P<id>{_APPROVAL_ID})", text, re.IGNORECASE
        )
        if approval:
            return self._match(
                "approval_command",
                "general_agent",
                "april-brain",
                3,
                "code_write",
                True,
                "Execute the referenced one-time approval only.",
                planned=[
                    PlannedToolCall(
                        tool="approve_action",
                        args={"approval_id": approval.group("id")},
                        reason="Dedicated exact approval command.",
                    )
                ],
                rule="approval.exact_id",
            )
        rejection = re.fullmatch(rf"(?:reject|deny)\s+(?P<id>{_APPROVAL_ID})", text, re.IGNORECASE)
        if rejection:
            return self._match(
                "rejection_command",
                "general_agent",
                "april-brain",
                0,
                "none",
                False,
                "Reject the referenced pending approval.",
                planned=[
                    PlannedToolCall(
                        tool="reject_action",
                        args={"approval_id": rejection.group("id")},
                        reason="Dedicated exact rejection command.",
                    )
                ],
                rule="approval.reject_exact_id",
            )

        git_patterns = (
            ("git_status", r"(?:show\s+)?(?:git|repository|repo)\s+status"),
            ("git_diff", r"(?:show\s+)?git\s+diff"),
            ("git_log", r"(?:show\s+)?git\s+log(?:\s+--oneline)?"),
            (
                "git_branch",
                r"(?:show\s+)?git\s+branch(?:es)?|(?:list|show)\s+(?:git\s+)?branches",
            ),
        )
        for tool, pattern in git_patterns:
            if re.fullmatch(pattern, lowered):
                return self._tool_match(
                    intent="coding_repo_analysis",
                    agent="coding_agent",
                    model_id="april-coding",
                    tool=tool,
                    args={},
                    level=1,
                    risk="read_only",
                    summary=f"Run the exact read-only {tool} inspection.",
                    rule=f"git.{tool.removeprefix('git_')}",
                )

        read = re.fullmatch(
            rf"(?:read|show|display)\s+(?:file\s+)?(?P<path>{_PATH})", text, re.IGNORECASE
        )
        if read and self._looks_like_exact_filename(read.group("path")):
            return self._tool_match(
                intent="document_reading",
                agent="reading_agent",
                model_id="april-reading",
                tool="read_file",
                args={"path": read.group("path")},
                level=1,
                risk="read_only",
                summary="Read the exact requested local file.",
                rule="file.read_exact_relative",
            )

        search = re.fullmatch(
            rf"(?:search|find)\s+(?:the\s+)?(?:repository|repo|files?)\s+for\s+"
            rf"(?P<query>[^/\\]{{1,200}}?)(?:\s+in\s+(?P<path>{_PATH}))?",
            text,
            re.IGNORECASE,
        )
        if search:
            query = search.group("query").strip(" \"'")
            if query:
                return self._tool_match(
                    intent="repository_search",
                    agent="coding_agent",
                    model_id="april-coding",
                    tool="search_files",
                    args={"path": search.group("path") or ".", "query": query, "limit": 20},
                    level=1,
                    risk="read_only",
                    summary="Search the selected local repository for the exact query.",
                    rule="repo.search_exact",
                )

        reminder_list = re.fullmatch(r"(?:list|show)\s+(?:my\s+)?reminders", lowered)
        if reminder_list:
            return self._tool_match(
                intent="reminder_list",
                agent="general_agent",
                model_id="april-brain",
                tool="list_reminders",
                args={},
                level=1,
                risk="read_only",
                summary="List local reminders.",
                rule="reminder.list",
            )
        reminder_cancel = re.fullmatch(
            rf"(?:cancel|delete)\s+reminder\s+(?P<id>{_APPROVAL_ID})",
            text,
            re.IGNORECASE,
        )
        if reminder_cancel:
            return self._tool_match(
                intent="reminder_cancel",
                agent="general_agent",
                model_id="april-brain",
                tool="cancel_reminder",
                args={"reminder_id": reminder_cancel.group("id")},
                level=2,
                risk="safe_write",
                summary="Cancel the identified local reminder.",
                rule="reminder.cancel_exact_id",
            )
        reminder_create = re.fullmatch(
            r"(?:create\s+(?:a\s+)?reminder(?:\s+to|\s+for)?|remind\s+me\s+to)\s+"
            r"(?P<content>.+)",
            text,
            re.IGNORECASE,
        )
        if reminder_create:
            content = reminder_create.group("content").strip()
            if content:
                return self._tool_match(
                    intent="reminder_create",
                    agent="general_agent",
                    model_id="april-brain",
                    tool="create_reminder",
                    args={"content": content},
                    level=2,
                    risk="safe_write",
                    summary="Create the exact local reminder.",
                    rule="reminder.create_explicit",
                )

        test = re.fullmatch(
            r"(?:run|execute)\s+(?:the\s+)?(?:configured\s+)?tests?"
            r"(?:\s+(?P<target>[A-Za-z0-9_./:-]+))?",
            text,
            re.IGNORECASE,
        )
        if test:
            args: dict[str, object] = {}
            if test.group("target"):
                args["argv"] = ["pytest", test.group("target")]
            return self._tool_match(
                intent="configured_test_execution",
                agent="coding_agent",
                model_id="april-coding",
                tool="test_runner",
                args=args,
                level=3,
                risk="code_write",
                summary="Run configured tests through exact-action approval.",
                rule="test.configured",
                confirmation=True,
            )

        if re.fullmatch(r"(?:propose|draft)\s+(?:a\s+)?patch(?:\s+for\s+.+)?", text, re.I):
            return self._match(
                "patch_proposal",
                "coding_agent",
                "april-coding",
                1,
                "read_only",
                False,
                "Prepare a read-only patch proposal without applying it.",
                tools=["git_status"],
                rule="patch.propose",
            )
        return None

    @staticmethod
    def _looks_like_exact_filename(path: str) -> bool:
        name = path.rsplit("/", maxsplit=1)[-1]
        return (
            "/" in path
            or "." in name.strip(".")
            or name.lower() in {"readme", "license", "makefile", "dockerfile", "agents.md"}
        )

    def _safety_route(self, text: str) -> DeterministicMatch | None:
        rules: tuple[tuple[str, str, int, RiskLevel, str], ...] = (
            (
                "safety.prompt_injection",
                r"(?:ignore|disregard)\s+(?:all\s+)?(?:previous|system|developer)\s+instructions"
                r"|reveal\s+(?:the\s+)?system\s+prompt",
                0,
                "none",
                "Prompt-injection text cannot override APRIL policy.",
            ),
            (
                "safety.path_escape",
                r".*(?:\.\./|/etc/passwd|\.ssh(?:/|\b)|id_rsa\b|keychain|browser profile).*",
                1,
                "read_only",
                "Sensitive or escaped filesystem paths are denied.",
            ),
            (
                "safety.package_install",
                r"(?:pip|npm|brew)\s+install(?:\s+.+)?|install\s+(?:a\s+)?package(?:\s+.+)?",
                5,
                "external_action",
                "Package installation is unsupported.",
            ),
            (
                "safety.external_action",
                r"(?:git\s+push|deploy(?:\s+.+)?|send\s+(?:an?\s+)?email(?:\s+.+)?"
                r"|publish(?:\s+.+)?|pay(?:ment|\s+.+))",
                5,
                "external_action",
                "External actions are disabled by policy.",
            ),
            (
                "safety.destructive",
                r"(?:rm\s+-rf|wipe|erase|delete)\s+(?:everything|all(?:\s+files)?|/)(?:\s+.*)?",
                4,
                "system_action",
                "Broad destructive actions are unsupported.",
            ),
            (
                "safety.unknown_tool",
                r"(?:use|run|call)\s+(?:the\s+)?(?:unknown|unsupported)\s+tool(?:\s+.+)?"
                r"|(?:use|run|call)\s+plasma_tool(?:\s+.*)?",
                0,
                "none",
                "Unknown tools are denied.",
            ),
        )
        for rule, pattern, level, risk, summary in rules:
            if re.fullmatch(pattern, text):
                agent: AgentName = "system_action_agent" if level >= 4 else "general_agent"
                return self._match(
                    rule.replace("safety.", ""),
                    agent,
                    "april-brain",
                    level,
                    risk,
                    level >= 3,
                    summary,
                    rule=rule,
                )
        return None

    def _tool_match(
        self,
        *,
        intent: str,
        agent: AgentName,
        model_id: str,
        tool: str,
        args: dict[str, object],
        level: int,
        risk: RiskLevel,
        summary: str,
        rule: str,
        confirmation: bool = False,
    ) -> DeterministicMatch:
        return self._match(
            intent,
            agent,
            model_id,
            level,
            risk,
            confirmation,
            summary,
            tools=[tool],
            planned=[
                PlannedToolCall(
                    tool=tool,
                    args=args,
                    reason="Bounded deterministic route.",
                )
            ],
            rule=rule,
        )

    def _match(
        self,
        intent: str,
        agent: AgentName,
        model_id: str,
        level: int,
        risk: RiskLevel,
        confirmation: bool,
        summary: str,
        *,
        tools: list[str] | None = None,
        planned: list[PlannedToolCall] | None = None,
        rule: str,
    ) -> DeterministicMatch:
        decision = BrainDecision(
            intent=intent,
            agent=agent,
            model_id=model_id,
            confidence=1.0,
            tools_needed=tools or [call.tool for call in planned or []],
            planned_tool_calls=planned or [],
            permission_level=level,
            risk_level=risk,
            needs_confirmation=confirmation,
            task_steps=[summary],
            decision_summary=summary,
            routing_method="fallback",
        )
        return DeterministicMatch(decision=decision, matched_rule=rule)
