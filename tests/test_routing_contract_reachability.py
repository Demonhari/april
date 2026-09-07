from __future__ import annotations

from pathlib import Path
from typing import get_args

from apps.runner.evals import _evaluate_case, load_brain_eval_cases
from services.brain.route_contract import (
    RouteCompiler,
    RouteContext,
    RouteOperation,
    RouteToolClass,
    RoutingProposal,
)


def test_every_routing_fixture_is_reachable_by_the_compiler() -> None:
    compiler = RouteCompiler.from_home(Path.cwd())
    operations = get_args(RouteOperation)
    contexts = get_args(RouteContext)
    tool_classes = get_args(RouteToolClass)
    for case in load_brain_eval_cases(Path.cwd()):
        reachable = False
        for operation in operations:
            for context in contexts:
                for tool_class in tool_classes:
                    try:
                        proposal = RoutingProposal(
                            operation=operation,
                            context=context,
                            tool_class=tool_class,
                            requested_text="x",
                            memory_queries=["q"],
                        )
                        decision = compiler.compile(proposal)
                    except (TypeError, ValueError, KeyError):
                        continue
                    actual = decision.model_dump()
                    actual.update(
                        route_source="model",
                        route_provenance="trusted_model_only_v1",
                    )
                    result = _evaluate_case(
                        case,
                        actual,
                        schema_valid=True,
                        allow_fallback=False,
                    )
                    if result.ok:
                        reachable = True
                        break
                if reachable:
                    break
            if reachable:
                break
        assert reachable, f"fixture case is unreachable: {case.id}"
