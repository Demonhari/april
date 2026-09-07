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

        memory_write = self._explicit_memory_write(text)
        if memory_write is not None:
            memory_type, content = memory_write
            return self._match(
                "memory_write",
                "general_agent",
                "april-brain",
                2,
                "safe_write",
                False,
                "Store the explicitly requested local durable memory.",
                tools=["remember_memory"],
                planned=[
                    PlannedToolCall(
                        tool="remember_memory",
                        args={
                            "content": content,
                            "memory_type": memory_type,
                            "reason": "Explicit user-requested durable local memory.",
                        },
                        reason="Store explicit local durable memory.",
                    )
                ],
                rule="memory.write_explicit",
            )

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
                tools=["git_status", "search_files"],
                rule="patch.propose",
            )
        return None

    def _explicit_memory_write(self, message: str) -> tuple[str, str] | None:
        """Recognize only an anchored, affirmative durable-memory command."""
        normalized = " ".join(message.strip().split())
        if (
            not normalized
            or normalized[0] in {'"', "'", "`"}
            or ord(normalized[0])
            in {
                0x2018,
                0x201C,
            }
        ):
            return None
        if re.match(r"^(?:for example|e\.g\.?|example:)\b", normalized, re.I):
            return None
        match = re.match(
            r"^(?:(?:please|could you|would you|can you)\s+|"
            r"i(?:'d| would) like you to\s+)?(?:april[, :]*)?"
            r"(?:remember|save|store|keep|note|make\s+a\s+note)"
            r"(?:\s+(?:that|this|as a memory))?\s+(.+)$",
            normalized,
            re.IGNORECASE,
        )
        if match is None:
            return None
        content = match.group(1).strip().strip("\"'`").strip()
        if not content or re.match(
            r"^(?:not|never|don(?:'t|t)|do not|no need to)\b", content, re.I
        ):
            return None
        lowered = content.casefold()
        if any(
            term in lowered for term in ("password", "secret", "token", "api key", "private key")
        ):
            return None
        if "prefer" in lowered or "preference" in lowered:
            return "preference", content
        if "project" in lowered:
            if re.search(r"\b(?:called|named|name is|known as)\b", lowered):
                return "relationship", content
            return "project_state", content
        return "fact", content

    @staticmethod
    def _looks_like_exact_filename(path: str) -> bool:
        name = path.rsplit("/", maxsplit=1)[-1]
        return (
            "/" in path
            or "." in name.strip(".")
            or name.lower() in {"readme", "license", "makefile", "dockerfile", "agents.md"}
        )

    def _safety_route(self, text: str) -> DeterministicMatch | None:
        question = re.match(
            r"^(?:how|what|why|when|which|where|explain|describe|tell me|is it|"
            r"can you explain|should i)\b",
            text,
        )
        action_guarded = bool(question)
        rules: tuple[tuple[str, str, AgentName, int, RiskLevel, bool, list[str], str], ...] = (
            (
                "prompt_injection",
                r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier|system|developer)\s+(?:instructions|prompts?|rules)\b|\breveal\s+(?:the\s+)?system\s+prompt\b",
                "general_agent",
                0,
                "none",
                False,
                [],
                "Prompt-injection text cannot override APRIL policy.",
            ),
            (
                "sensitive_content",
                r"\bsk-[a-z0-9_-]{12,}\b|\b(?:api[ _-]?key|password|passphrase|"
                r"secret|access token|auth token|private key)\s*(?:is|=|:)\s*\S",
                "general_agent",
                0,
                "none",
                False,
                [],
                "Handle sensitive content without taking an unsafe action.",
            ),
            (
                "path_escape_attempt",
                r"\.\./|/etc/passwd|(?:^|[\s/])\.ssh(?:/|\b)|\bid_rsa\b|"
                r"\bkeychain\b|\bbrowser profile\b",
                "general_agent",
                1,
                "read_only",
                False,
                [],
                "Sensitive or escaped filesystem paths are denied.",
            ),
            (
                "package_install",
                r"\b(?:pip3?|npm|pnpm|yarn|brew|apt(?:-get)?|cargo|gem)\s+install\b|\binstall\b.{0,60}\b(?:with|via|using)\s+(?:pip3?|npm|brew|apt)\b",
                "system_action_agent",
                5,
                "external_action",
                True,
                [],
                "Package installation requires explicit approval.",
            ),
            (
                "external_action",
                r"\bgit\s+push\b|^(?:please\s+)?push\b.{0,60}\b(?:branch|commits?|"
                r"github|origin|remote)\b|^(?:please\s+)?deploy\b|\band\s+deploy\b|"
                r"\bsend\s+(?:an?\s+|the\s+)?(?:e-?mail|message|sms|text message)\b|"
                r"^(?:please\s+)?e-?mail\s+\w+|\bpay\s+(?:the\s+|my\s+)?"
                r"(?:invoice|bill)\b|\bmake\s+(?:a\s+)?payment\b|\band\s+publish\b|"
                r"^(?:please\s+)?publish\b",
                "system_action_agent",
                5,
                "external_action",
                True,
                [],
                "External actions require explicit approval.",
            ),
            (
                "log_cleanup",
                r"^(?:please\s+)?(?:delete|remove|clear|clean(?:\s+up)?|purge)\b.{0,40}\blogs?\b",
                "system_action_agent",
                4,
                "system_action",
                True,
                ["plan_log_cleanup"],
                "Plan scoped local log cleanup for approval.",
            ),
            (
                "command_execution",
                r"^(?:please\s+)?(?:run|execute)\s+(?:pytest|the\s+tests?|tests?|the\s+test\s+suite|ruff|mypy|make\b)",
                "system_action_agent",
                3,
                "code_write",
                True,
                ["run_command"],
                "Run the configured command through approval.",
            ),
            (
                "unsupported_tool",
                r"\b(?:unknown|unsupported)\s+(?:tool|\w+_tool)\b|\b(?:use|run|call|invoke)\s+(?:the\s+)?\w+_tool\b",
                "general_agent",
                0,
                "none",
                False,
                [],
                "Unknown tools are denied.",
            ),
        )
        for name, pattern, agent, level, risk, confirmation, tools, summary in rules:
            if action_guarded and name in {
                "package_install",
                "external_action",
                "log_cleanup",
                "command_execution",
            }:
                continue
            if re.search(pattern, text):
                return self._match(
                    name,
                    agent,
                    "april-brain",
                    level,
                    risk,
                    confirmation,
                    summary,
                    tools=tools,
                    rule=f"safety.{name}",
                )
        if re.search(
            r"(?:rm\s+-rf|wipe|erase|delete)\s+(?:everything|all(?:\s+files)?|/)(?:\s+.*)?",
            text,
        ):
            return self._match(
                "destructive_action",
                "system_action_agent",
                "april-brain",
                4,
                "system_action",
                True,
                "Broad destructive actions are unsupported.",
                rule="safety.destructive",
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
            tools=["git_status", "search_files"] if tool == "git_status" else [tool],
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
