from __future__ import annotations

import re

from services.brain.schemas import BrainDecision
from services.permissions.schemas import RiskLevel


class FallbackRouter:
    def route(self, message: str) -> BrainDecision:
        normalized = message.lower()
        if self._contains(normalized, "delete", "remove old logs", "wipe"):
            return self._decision(
                intent="destructive_action",
                agent="system_action_agent",
                model_id="april-brain",
                tools=["list_files"],
                level=4,
                risk="system_action",
                confirmation=True,
                summary="Destructive local action requires explicit approval.",
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
        if self._contains(normalized, "apply the fix", "edit", "modify", "write code", "fix it"):
            return self._decision(
                intent="code_modification",
                agent="coding_agent",
                model_id="april-coding",
                tools=["patch_applier"],
                level=3,
                risk="code_write",
                confirmation=True,
                summary="Code modification requires exact approval.",
            )
        if self._contains(normalized, "repo", "repository", "animation", "bug", "code", "why"):
            return self._decision(
                intent="coding_repo_analysis",
                agent="coding_agent",
                model_id="april-coding",
                tools=["git_status", "search_files", "read_file"],
                level=1,
                risk="read_only",
                confirmation=False,
                summary="Read-only repository investigation.",
            )
        if self._contains(normalized, "read", "summarize", "document", "pdf", "file"):
            return self._decision(
                intent="document_reading",
                agent="reading_agent",
                model_id="april-reading",
                tools=["read_file"],
                level=1,
                risk="read_only",
                confirmation=False,
                summary="Document reading request.",
            )
        if self._contains(normalized, "remember", "memory", "recall"):
            return self._decision(
                intent="memory_lookup",
                agent="general_agent",
                model_id="april-brain",
                tools=[],
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

    def _decision(
        self,
        *,
        intent: str,
        agent: str,
        model_id: str,
        tools: list[str],
        level: int,
        risk: RiskLevel,
        confirmation: bool,
        summary: str,
    ) -> BrainDecision:
        return BrainDecision(
            intent=intent,
            agent=agent,
            model_id=model_id,
            tools_needed=tools,
            memory_queries=[],
            permission_level=level,
            risk_level=risk,
            needs_confirmation=confirmation,
            task_steps=[summary],
            decision_summary=summary,
            routing_method="fallback",
        )
