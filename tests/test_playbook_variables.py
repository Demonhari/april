from __future__ import annotations

import pytest

from skills.playbooks import PlaybookDefinition, PlaybookRunner
from skills.playbooks.variables import (
    MAX_EACH_PROJECTS,
    PlaybookExpansionError,
    expand_playbook_steps,
)
from tests.test_playbooks import _tool_executor


def _definition(steps: list[dict]) -> PlaybookDefinition:
    return PlaybookDefinition(
        id="vars-playbook",
        name="Variables playbook",
        agent_id="playbook_agent",
        status="active",
        trigger_examples=["run vars playbook"],
        steps=steps,
    )


@pytest.mark.asyncio
async def test_last_run_substitutes_none_then_real_summary(settings_tmp) -> None:
    database, memory, executor = await _tool_executor(settings_tmp)
    try:
        playbook = _definition(
            [{"tool": "echo", "args": {"value": "last run was: $last_run"}}]
        )
        runner = PlaybookRunner(executor, memory=memory)
        first = await runner.run(playbook)
        assert first.status == "completed"
        assert first.steps[0].result is not None
        first_echo = first.steps[0].result["data"]["echo"]["value"]
        assert first_echo == "last run was: none"

        second = await runner.run(playbook)
        second_echo = second.steps[0].result["data"]["echo"]["value"]
        assert "status=completed" in second_echo
        assert "steps_completed=1" in second_echo
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_each_active_projects_expands_deterministically(settings_tmp) -> None:
    database, memory, executor = await _tool_executor(settings_tmp)
    try:
        (settings_tmp.home / "a").mkdir()
        (settings_tmp.home / "b").mkdir()
        project_a = await memory.add_project(str(settings_tmp.home / "a"), name="a")
        project_b = await memory.add_project(str(settings_tmp.home / "b"), name="b")
        playbook = _definition(
            [{"tool": "echo", "args": {"value": "check $each(active_projects)"}}]
        )
        result = await PlaybookRunner(executor, memory=memory).run(playbook)
        assert result.status == "completed"
        assert result.steps_completed == 2
        values = [step.result["data"]["echo"]["value"] for step in result.steps]
        assert values == [f"check {project_a.path}", f"check {project_b.path}"]
        project_ids = [step.result["data"]["echo"]["project_id"] for step in result.steps]
        assert project_ids == [project_a.id, project_b.id]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_each_with_no_projects_expands_to_zero_steps(settings_tmp) -> None:
    database, memory, executor = await _tool_executor(settings_tmp)
    try:
        playbook = _definition(
            [
                {"tool": "echo", "args": {"value": "per-project $each(active_projects)"}},
                {"tool": "echo", "args": {"value": "always runs"}},
            ]
        )
        result = await PlaybookRunner(executor, memory=memory).run(playbook)
        assert result.status == "completed"
        assert result.steps_completed == 1
        assert result.steps[0].result["data"]["echo"]["value"] == "always runs"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_each_expansion_is_bounded(settings_tmp) -> None:
    database, memory, _executor = await _tool_executor(settings_tmp)
    try:
        for index in range(MAX_EACH_PROJECTS + 3):
            await memory.add_project(str(settings_tmp.home / f"p{index}"), name=f"p{index}")
        playbook = _definition(
            [{"tool": "echo", "args": {"value": "check $each(active_projects)"}}]
        )
        expansion = await expand_playbook_steps(playbook, memory=memory)
        assert len(expansion.steps) == MAX_EACH_PROJECTS
        assert any("capped" in note for note in expansion.notes)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_expansion_over_total_step_budget_fails_closed(settings_tmp) -> None:
    database, memory, executor = await _tool_executor(settings_tmp)
    try:
        for index in range(8):
            await memory.add_project(str(settings_tmp.home / f"p{index}"), name=f"p{index}")
        # 6 $each steps x 8 projects = 48 expanded steps > 40 budget.
        playbook = _definition(
            [
                {"tool": "echo", "args": {"value": f"pass {step} $each(active_projects)"}}
                for step in range(6)
            ]
        )
        with pytest.raises(PlaybookExpansionError):
            await expand_playbook_steps(playbook, memory=memory)
        result = await PlaybookRunner(executor, memory=memory).run(playbook)
        assert result.status == "failed"
        assert result.steps_completed == 0
        runs = await memory.list_playbook_runs(playbook_id="vars-playbook")
        assert "maximum" in (runs[0]["detail"] or "")
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_expanded_l3_steps_still_require_exact_approval(settings_tmp) -> None:
    database, memory, executor = await _tool_executor(settings_tmp)
    try:
        (settings_tmp.home / "a").mkdir()
        (settings_tmp.home / "b").mkdir()
        await memory.add_project(str(settings_tmp.home / "a"), name="a")
        await memory.add_project(str(settings_tmp.home / "b"), name="b")
        playbook = _definition(
            [{"tool": "dangerous", "args": {"value": "touch $each(active_projects)"}}]
        )
        result = await PlaybookRunner(executor, memory=memory).run(playbook)
        # The very first expanded step pauses for approval; nothing executed.
        assert result.status == "pending_approval"
        assert result.steps_completed == 0
        assert result.steps[0].approval is not None
        pending = await database.fetchall("SELECT * FROM approvals WHERE status = 'pending'")
        assert len(pending) == 1
        executed = await database.fetchall(
            "SELECT * FROM tool_calls WHERE tool = 'dangerous' AND status = 'executed'"
        )
        assert executed == []
    finally:
        await database.close()
