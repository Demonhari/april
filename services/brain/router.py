from __future__ import annotations

from april_common.errors import AprilError
from services.april_runtime.client import RuntimeClient
from services.brain.deterministic_router import DeterministicRouter
from services.brain.fallback_router import FallbackRouter
from services.brain.model_routing import infer_model_route, routing_contract_fingerprint
from services.brain.route_contract import (
    RouteCompiler,
    build_router_system_prompt,
    default_route_compiler,
)
from services.brain.schemas import BrainDecision, RouteResult, RouteSource
from services.memory.schemas import Message

ROUTER_SYSTEM_PROMPT = build_router_system_prompt()


class BrainRouter:
    def __init__(
        self,
        runtime_client: RuntimeClient,
        *,
        brain_model_id: str = "april-brain",
        router_model_id: str | None = None,
        deterministic_router: DeterministicRouter | None = None,
        route_compiler: RouteCompiler | None = None,
    ) -> None:
        self.runtime_client = runtime_client
        self.brain_model_id = brain_model_id
        self.router_model_id = router_model_id or brain_model_id
        self.deterministic = deterministic_router or DeterministicRouter()
        self.fallback = FallbackRouter()
        self.route_compiler = route_compiler or default_route_compiler()
        self.router_system_prompt = build_router_system_prompt(self.route_compiler.bindings)

    async def route(
        self,
        message: str,
        *,
        request_id: str | None = None,
        history: list[Message] | None = None,
    ) -> BrainDecision:
        return (
            await self.route_result(
                message,
                request_id=request_id,
                history=history,
            )
        ).decision

    async def route_result(
        self,
        message: str,
        *,
        request_id: str | None = None,
        history: list[Message] | None = None,
    ) -> RouteResult:
        deterministic = self.deterministic.route(message)
        if deterministic is not None:
            return RouteResult(
                decision=deterministic.decision,
                route_source=RouteSource.DETERMINISTIC,
                effective_confidence=1.0,
                confidence_source="deterministic_rule",
                matched_rule=deterministic.matched_rule,
            )

        try:
            outcome = await infer_model_route(
                self.runtime_client,
                model_id=self.router_model_id,
                message=message,
                history=history,
                request_id=request_id,
                compiler=self.route_compiler,
                system_prompt=self.router_system_prompt,
                max_output_tokens=192,
            )
            if outcome.decision is None:
                return self._fallback_result(
                    message,
                    reason=outcome.failure_code or "runtime_or_output_failure",
                    outcome=outcome,
                )
            source = outcome.route_source or RouteSource.MODEL
            return RouteResult(
                decision=outcome.decision,
                route_source=source,
                raw_model_confidence=outcome.proposal.confidence if outcome.proposal else None,
                effective_confidence=outcome.decision.confidence,
                confidence_source="raw_model_proposal",
                structured_output_valid=True,
                repair_used=outcome.repair_attempted,
                proposal_operation=outcome.proposal.operation if outcome.proposal else None,
                proposal_context=outcome.proposal.context if outcome.proposal else None,
                contract_fingerprint=routing_contract_fingerprint(self.route_compiler),
                repair_attempted=outcome.repair_attempted,
                repair_succeeded=outcome.repair_succeeded,
                routing_failure_code=outcome.failure_code,
            )
        except (AprilError, TimeoutError, OSError):
            return self._fallback_result(message, reason="runtime_or_output_failure")

    def _fallback_result(
        self, message: str, *, reason: str, outcome: object | None = None
    ) -> RouteResult:
        decision = self.fallback.route(message)
        repair_attempted = bool(getattr(outcome, "repair_attempted", False))
        repair_succeeded = bool(getattr(outcome, "repair_succeeded", False))
        failure_code = getattr(outcome, "failure_code", None)
        return RouteResult(
            decision=decision,
            route_source=RouteSource.FALLBACK,
            effective_confidence=decision.confidence,
            confidence_source="fallback_policy",
            fallback_reason=reason,
            structured_output_valid=False,
            repair_attempted=repair_attempted,
            repair_succeeded=repair_succeeded,
            routing_failure_code=failure_code or reason,
            contract_fingerprint=routing_contract_fingerprint(self.route_compiler),
        )
