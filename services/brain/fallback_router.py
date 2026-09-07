from __future__ import annotations

import re

from agents.schemas import AgentName
from services.brain.schemas import BrainDecision, PlannedToolCall
from services.permissions.schemas import RiskLevel


class FallbackRouter:
    def route(self, message: str) -> BrainDecision:
        normalized = message.lower()
        if self._looks_like_prompt_injection(normalized):
            return self._decision(
                intent="prompt_injection",
                agent="general_agent",
                model_id="april-brain",
                tools=[],
                level=0,
                risk="none",
                confirmation=False,
                summary="Treat untrusted instructions as user text.",
            )
        if self._looks_like_path_escape(normalized):
            return self._decision(
                intent="path_escape_attempt",
                agent="general_agent",
                model_id="april-brain",
                tools=[],
                level=1,
                risk="read_only",
                confirmation=False,
                summary="Refuse access outside configured local roots.",
            )
        if self._contains(normalized, "unknown tool", "unsupported tool", "plasma_tool"):
            return self._decision(
                intent="unsupported_tool",
                agent="general_agent",
                model_id="april-brain",
                tools=[],
                level=0,
                risk="none",
                confirmation=False,
                summary="Unknown tools are denied.",
            )
        if self._contains(
            normalized,
            "install package",
            "install a package",
            "install the",
            "pip install",
            "npm install",
            "brew install",
        ):
            return self._decision(
                intent="package_install",
                agent="system_action_agent",
                model_id="april-brain",
                tools=[],
                level=5,
                risk="external_action",
                confirmation=True,
                summary="Package installation is outside the MVP.",
            )
        if self._contains(
            normalized,
            "delete old logs",
            "delete the old logs",
            "delete logs",
            "remove old logs",
            "remove logs",
            "clear old logs",
            "clear logs",
            "clean up logs",
            "clean up old logs",
            "clean logs",
            "old log files",
            "purge logs",
            "delete old audio",
            "clear audio cache",
            "clean up audio cache",
        ):
            # Scoped, two-stage cleanup: planning is read-only and produces an
            # immutable manifest; applying it is a Level 4 system action that
            # still requires exact approval. The engine remains authoritative.
            return self._decision(
                intent="log_cleanup",
                agent="system_action_agent",
                model_id="april-brain",
                tools=["plan_log_cleanup"],
                level=4,
                risk="system_action",
                confirmation=True,
                summary=(
                    "Plan local log cleanup first (read-only); "
                    "applying deletions requires exact approval."
                ),
            )
        if self._contains(normalized, "delete", "wipe", "rm -rf", "erase everything"):
            # APRIL has no generic/recursive delete tool by design. Only the
            # scoped log/audio-cache cleanup flow above can remove files.
            return self._decision(
                intent="destructive_action",
                agent="system_action_agent",
                model_id="april-brain",
                tools=[],
                level=4,
                risk="system_action",
                confirmation=True,
                summary="Broad deletion is unsupported; only scoped log cleanup is available.",
            )
        if self._contains(normalized, "push", "deploy", "send email", "payment", "publish"):
            return self._decision(
                intent="external_action",
                agent="system_action_agent",
                model_id="april-brain",
                tools=[],
                level=5,
                risk="external_action",
                confirmation=True,
                summary="External action is not enabled in the MVP.",
            )
        if self._contains(normalized, "pay the invoice", "pay invoice", "using my card"):
            return self._decision(
                intent="external_action",
                agent="system_action_agent",
                model_id="april-brain",
                tools=[],
                level=5,
                risk="external_action",
                confirmation=True,
                summary="External action is not enabled in the MVP.",
            )
        if self._contains(
            normalized,
            "run pytest",
            "execute command",
            "shell command",
            "terminal command",
        ):
            return self._decision(
                intent="command_execution",
                agent="system_action_agent",
                model_id="april-brain",
                tools=["run_command"],
                level=3,
                risk="code_write",
                confirmation=True,
                summary="Configured local command execution requires approval.",
            )
        if self._contains(normalized, "propose a patch", "draft a patch", "patch proposal"):
            return self._decision(
                intent="patch_proposal",
                agent="coding_agent",
                model_id="april-coding",
                tools=["git_status", "search_files"],
                level=1,
                risk="read_only",
                confirmation=False,
                summary="Draft a patch proposal without applying changes.",
            )
        if self._contains(
            normalized,
            "apply the fix",
            "apply the patch",
            "apply patch",
            "edit",
            "modify",
            "write code",
            "fix it",
            "fix this",
        ):
            return self._decision(
                intent="code_modification",
                agent="coding_agent",
                model_id="april-coding",
                tools=[],
                level=3,
                risk="code_write",
                confirmation=True,
                summary="Code modification requires exact approval.",
            )
        memory_write = self._explicit_memory_write(message)
        if memory_write is not None:
            memory_type, content = memory_write
            return self._decision(
                intent="memory_write",
                agent="general_agent",
                model_id="april-brain",
                tools=["remember_memory"],
                planned_tool_calls=[
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
                level=2,
                risk="safe_write",
                confirmation=False,
                summary="Explicit durable local memory write.",
            )
        if self._looks_like_tool_free_coding(normalized):
            return self._decision(
                intent="coding_assistance",
                agent="coding_agent",
                model_id="april-coding",
                tools=[],
                level=0,
                risk="none",
                confirmation=False,
                summary="Answer the supplied code without accessing a repository.",
            )
        git_tools = self._git_read_tools(normalized)
        if git_tools:
            return self._decision(
                intent="coding_repo_analysis",
                agent="coding_agent",
                model_id="april-coding",
                tools=git_tools,
                level=1,
                risk="read_only",
                confirmation=False,
                summary="Read-only Git repository request.",
            )
        if self._contains(
            normalized, "repo", "repository", "animation", "bug", "inspect", "search files"
        ):
            return self._decision(
                intent="coding_repo_analysis",
                agent="coding_agent",
                model_id="april-coding",
                tools=["git_status", "search_files"],
                level=1,
                risk="read_only",
                confirmation=False,
                summary="Read-only repository investigation.",
            )
        if self._contains(
            normalized,
            "think deeply",
            "deep dive",
            "reason through",
            "trade-off",
            "trade-offs",
            "tradeoff",
            "trade offs",
            "pros and cons",
            "design decision",
            "weigh the options",
            "evaluate options",
            "compare approaches",
            "architectural decision",
        ):
            return self._decision(
                intent="deep_reasoning",
                agent="reasoning_agent",
                model_id="april-brain",
                tools=[],
                level=1,
                risk="read_only",
                confirmation=False,
                summary="Deep reasoning and architecture analysis (read-only).",
            )
        if self._contains(normalized, "read", "summarize", "document", "pdf", "file"):
            return self._decision(
                intent="document_reading",
                agent="reading_agent",
                model_id="april-reading",
                tools=[],
                level=1,
                risk="read_only",
                confirmation=False,
                summary="Document reading request.",
            )
        if self._looks_like_secret(normalized):
            return self._decision(
                intent="sensitive_content",
                agent="general_agent",
                model_id="april-brain",
                tools=[],
                level=0,
                risk="none",
                confirmation=False,
                summary="Avoid storing or exposing secret-like content.",
            )
        recall = self._recall_subject(message)
        if recall is not None or self._contains(normalized, "memory", "recall"):
            return self._decision(
                intent="memory_lookup",
                agent="general_agent",
                model_id="april-brain",
                tools=[],
                memory_queries=[recall] if recall is not None else [],
                level=0,
                risk="none",
                confirmation=False,
                summary="Memory lookup or durable memory request.",
            )
        if self._contains(normalized, "remind me", "reminder"):
            return self._decision(
                intent="reminders",
                agent="general_agent",
                model_id="april-brain",
                tools=["create_reminder"],
                planned_tool_calls=[
                    PlannedToolCall(
                        tool="create_reminder",
                        args={"content": message},
                        reason="Create local reminder from explicit reminder request.",
                    )
                ],
                level=2,
                risk="safe_write",
                confirmation=False,
                summary="Local reminder request.",
            )
        if self._contains(normalized, "story", "email", "script", "idea", "creative"):
            return self._decision(
                intent="creative_writing",
                agent="creative_agent",
                model_id="april-brain",
                tools=[],
                level=0,
                risk="none",
                confirmation=False,
                summary="Creative writing request.",
            )
        if self._contains(normalized, "do the thing", "handle it", "make it happen"):
            return self._decision(
                intent="ambiguous_request",
                agent="general_agent",
                model_id="april-brain",
                tools=[],
                level=0,
                risk="none",
                confirmation=False,
                summary="Ambiguous request should be clarified before action.",
            )
        if self._contains(normalized, "plan", "schedule", "today", "strategy"):
            return self._decision(
                intent="planning",
                agent="general_agent",
                model_id="april-brain",
                tools=[],
                level=0,
                risk="none",
                confirmation=False,
                summary="Planning conversation.",
            )
        return self._decision(
            intent="normal_conversation",
            agent="general_agent",
            model_id="april-brain",
            tools=[],
            level=0,
            risk="none",
            confirmation=False,
            summary="General conversation.",
        )

    def _contains(self, text: str, *terms: str) -> bool:
        return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)

    def _looks_like_prompt_injection(self, text: str) -> bool:
        return self._contains(
            text,
            "ignore previous instructions",
            "disregard system",
            "reveal the system prompt",
            "developer message says",
        )

    def _looks_like_path_escape(self, text: str) -> bool:
        sensitive_terms = (".ssh", "id_rsa", "keychain", "/etc/passwd", "browser profile")
        return "../" in text or any(term in text for term in sensitive_terms)

    def _looks_like_secret(self, text: str) -> bool:
        return bool(
            re.search(r"\bsk-[a-z0-9_-]{12,}\b", text, flags=re.IGNORECASE)
            or re.search(r"\b(api key|password|private key|token)\b", text)
        )

    def _looks_like_tool_free_coding(self, text: str) -> bool:
        if self._contains(
            text,
            "repository",
            "repo",
            "codebase",
            "project",
            "file",
            "directory",
            "path",
            "git",
            "test",
            "pytest",
            "patch",
            "run",
            "execute",
            "inspect",
            "search",
        ):
            return False
        code_terms = (
            "code",
            "python",
            "javascript",
            "typescript",
            "snippet",
            "function",
            "class",
        )
        return self._contains(text, *code_terms) and bool(
            re.search(r"\b(?:write|create|explain|understand|what does|how does)\b", text)
        )

    def _git_read_tools(self, text: str) -> list[str]:
        tools: list[str] = []
        if self._contains(
            text,
            "git log",
            "show git log",
            "recent commits",
            "latest commits",
            "last commits",
            "commit history",
        ):
            tools.append("git_log")
        if self._contains(
            text,
            "git branch",
            "current branch",
            "list branches",
            "which branch",
            "show branches",
            "all branches",
        ):
            tools.append("git_branch")
        if self._contains(
            text,
            "git diff",
            "show git diff",
            "what changed",
            "what has changed",
            "unstaged changes",
            "uncommitted changes",
        ):
            tools.append("git_diff")
        if self._contains(
            text,
            "git status",
            "show git status",
            "working tree status",
            "repo status",
            "repository status",
        ):
            tools.append("git_status")
        if tools == ["git_status"]:
            return ["git_status", "search_files"]
        return tools

    def _explicit_memory_write(self, message: str) -> tuple[str, str] | None:
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

    def _recall_subject(self, message: str) -> str | None:
        match = re.match(
            r"^(?:what is|what's|whats)\s+(?:(?:the )?name of\s+)?my\s+(.+?)\??$",
            " ".join(message.strip().split()),
            re.IGNORECASE,
        )
        return match.group(1).strip() if match is not None else None

    def _decision(
        self,
        *,
        intent: str,
        agent: AgentName,
        model_id: str,
        tools: list[str],
        planned_tool_calls: list[PlannedToolCall] | None = None,
        memory_queries: list[str] | None = None,
        level: int,
        risk: RiskLevel,
        confirmation: bool,
        summary: str,
    ) -> BrainDecision:
        return BrainDecision(
            intent=intent,
            agent=agent,
            model_id=model_id,
            confidence=0.45,
            tools_needed=tools,
            planned_tool_calls=planned_tool_calls or [],
            memory_queries=memory_queries or [],
            permission_level=level,
            risk_level=risk,
            needs_confirmation=confirmation,
            task_steps=[summary],
            decision_summary=summary,
            routing_method="fallback",
        )
