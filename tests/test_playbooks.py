from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from april_common.audit import AuditLogger
from services.api.server import create_app
from services.evolution.playbook_miner import mine_playbook_candidates
from services.evolution.write_guard import EvolutionWriteGuard
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.permissions.approvals import ApprovalStore
from services.permissions.engine import PermissionEngine
from services.permissions.tool_execution import ToolExecutionService
from skills.playbooks import PlaybookDefinition, PlaybookLoader, PlaybookMiner, PlaybookRunner
from skills.registry import ToolRegistry
from skills.schemas import ToolDefinition, ToolResult
from tests.test_core_api import auth, make_container


async def _echo(args: dict[str, object]) -> ToolResult:
    return ToolResult(
        ok=True,
        data={"echo": args},
        risk_level="read_only",
        permission_level=1,
    )


async def _dangerous(args: dict[str, object]) -> ToolResult:
    return ToolResult(
        ok=True,
        data={"dangerous": args},
        risk_level="code_write",
        permission_level=3,
    )


async def _tool_executor(settings_tmp) -> tuple[Database, SqliteMemory, ToolExecutionService]:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="test echo",
            permission_level=1,
            risk_level="read_only",
            allowed_agents={"playbook_agent"},
            executor=_echo,
        )
    )
    registry.register(
        ToolDefinition(
            name="dangerous",
            description="test dangerous",
            permission_level=3,
            risk_level="code_write",
            confirmation_required=True,
            allowed_agents={"playbook_agent"},
            executor=_dangerous,
        )
    )
    approvals = ApprovalStore(database, AuditLogger(settings_tmp.audit_path), expiry_seconds=60)
    return (
        database,
        memory,
        ToolExecutionService(
            settings=settings_tmp,
            memory=memory,
            tool_registry=registry,
            permission_engine=PermissionEngine(registry),
            approvals=approvals,
        ),
    )


def _playbook(*tools: str) -> PlaybookDefinition:
    return PlaybookDefinition(
        id="test-playbook",
        name="Test playbook",
        agent_id="playbook_agent",
        status="active",
        trigger_examples=["run test playbook"],
        steps=[{"tool": tool, "args": {"value": tool}} for tool in tools],
    )


@pytest.mark.asyncio
async def test_playbook_runner_executes_safe_steps(settings_tmp) -> None:
    database, memory, executor = await _tool_executor(settings_tmp)
    try:
        result = await PlaybookRunner(executor, memory=memory).run(_playbook("echo"))
        assert result.status == "completed"
        assert result.steps_completed == 1
        rows = await database.fetchall("SELECT * FROM tool_calls WHERE tool = ?", ("echo",))
        assert len(rows) == 1
        # The run is persisted into playbook_runs.
        runs = await memory.list_playbook_runs(playbook_id="test-playbook")
        assert len(runs) == 1
        assert runs[0]["status"] == "completed"
        assert runs[0]["steps_completed"] == 1
        assert runs[0]["completed_at"] is not None
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_playbook_runner_pauses_for_l3_approval(settings_tmp) -> None:
    database, memory, executor = await _tool_executor(settings_tmp)
    try:
        result = await PlaybookRunner(executor, memory=memory).run(_playbook("dangerous"))
        assert result.status == "pending_approval"
        assert result.steps_completed == 0
        assert result.steps[0].approval is not None
        rows = await database.fetchall("SELECT * FROM approvals WHERE status = 'pending'")
        assert len(rows) == 1
        runs = await memory.list_playbook_runs(playbook_id="test-playbook")
        assert len(runs) == 1
        assert runs[0]["status"] == "pending_approval"
        assert "exact-action approval" in (runs[0]["detail"] or "")
    finally:
        await database.close()


def test_playbook_loader_ambiguous_trigger_falls_back(settings_tmp) -> None:
    loader = PlaybookLoader(settings_tmp.playbooks_path)
    first = _playbook("echo")
    second = first.model_copy(update={"id": "other-playbook", "name": "Other"})
    loader.adopt(first)
    loader.adopt(second)
    assert loader.match_trigger("please run test playbook") is None


def test_playbook_miner_uses_successful_tool_call_sequences() -> None:
    candidate = PlaybookMiner().mine(
        [
            {"tool": "search_files", "args": {"query": "TODO"}, "status": "executed"},
            {"tool": "read_file", "args": {"path": "README.md"}, "status": "executed"},
            {"tool": "write_file", "args": {"path": "x"}, "status": "failed"},
        ],
        trigger="inspect todos",
    )
    assert candidate is not None
    assert [step.tool for step in candidate.steps] == ["search_files", "read_file"]
    assert candidate.status == "candidate"


def test_playbook_miner_requires_frequent_sequences() -> None:
    sequences = [
        [
            {"tool": "search_files", "args": {"query": "TODO"}, "status": "executed"},
            {"tool": "read_file", "args": {"path": "README.md"}, "status": "executed"},
        ],
        [
            {"tool": "search_files", "args": {"query": "TODO"}, "status": "executed"},
            {"tool": "read_file", "args": {"path": "README.md"}, "status": "executed"},
        ],
        [
            {"tool": "search_files", "args": {"query": "TODO"}, "status": "executed"},
            {"tool": "read_file", "args": {"path": "README.md"}, "status": "executed"},
        ],
    ]
    candidates = PlaybookMiner().mine_frequent(
        sequences,
        support_threshold=3,
        known_tools={"search_files", "read_file"},
    )
    assert len(candidates) == 1
    assert [step.tool for step in candidates[0].steps] == ["search_files", "read_file"]
    assert "support=3" in candidates[0].steps[0].reason


async def _seed_tool_sequence(
    memory: SqliteMemory,
    sequence: list[tuple[str, dict[str, object]]],
) -> str:
    conversation_id = await memory.create_conversation()
    for tool, args in sequence:
        await memory.record_tool_call(
            tool=tool,
            args=args,
            status="executed",
            permission_level=1,
            risk_level="read_only",
            result={"ok": True},
            conversation_id=conversation_id,
        )
    return conversation_id


@pytest.mark.asyncio
async def test_playbook_mining_service_emits_supported_candidate(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    try:
        sequence = [
            ("search_files", {"query": "TODO"}),
            ("read_file", {"path": "README.md"}),
        ]
        for _ in range(3):
            await _seed_tool_sequence(memory, sequence)

        report = await mine_playbook_candidates(
            memory,
            settings_tmp,
            guard=EvolutionWriteGuard(settings_tmp),
            support_threshold=3,
        )

        assert len(report.candidate_ids) == 1
        candidate_path = settings_tmp.playbooks_path / f"{report.candidate_ids[0]}.yaml"
        assert candidate_path.exists()
        loaded = PlaybookLoader(settings_tmp.playbooks_path).get(report.candidate_ids[0])
        assert loaded is not None
        assert loaded.status == "candidate"
        assert [step.tool for step in loaded.steps] == ["search_files", "read_file"]
        assert report.support[loaded.id] == 3
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_playbook_mining_service_requires_support_threshold(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    try:
        sequence = [
            ("search_files", {"query": "TODO"}),
            ("read_file", {"path": "README.md"}),
        ]
        for _ in range(2):
            await _seed_tool_sequence(memory, sequence)

        report = await mine_playbook_candidates(
            memory,
            settings_tmp,
            guard=EvolutionWriteGuard(settings_tmp),
            support_threshold=3,
        )

        assert report.candidate_ids == []
        assert list(settings_tmp.playbooks_path.glob("mined-*.yaml")) == []
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_playbook_mining_skips_unknown_tools(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    try:
        sequence = [
            ("unknown_tool", {"query": "TODO"}),
            ("read_file", {"path": "README.md"}),
        ]
        for _ in range(3):
            await _seed_tool_sequence(memory, sequence)

        report = await mine_playbook_candidates(
            memory,
            settings_tmp,
            guard=EvolutionWriteGuard(settings_tmp),
            support_threshold=3,
        )

        assert report.candidate_ids == []
    finally:
        await database.close()


def test_playbook_miner_finds_frequent_contiguous_subsequences() -> None:
    # The shared two-step flow is embedded inside three longer, otherwise
    # different sequences; whole-sequence grouping would find nothing.
    core = [
        {"tool": "search_files", "args": {"query": "TODO"}, "status": "executed"},
        {"tool": "read_file", "args": {"path": "README.md"}, "status": "executed"},
    ]
    sequences = [
        [{"tool": "git_status", "args": {}, "status": "executed"}, *core],
        [*core, {"tool": "git_log", "args": {}, "status": "executed"}],
        [
            {"tool": "list_files", "args": {"path": "."}, "status": "executed"},
            *core,
            {"tool": "git_diff", "args": {}, "status": "executed"},
        ],
    ]
    candidates = PlaybookMiner().mine_frequent(
        sequences,
        support_threshold=3,
        known_tools={
            "search_files",
            "read_file",
            "git_status",
            "git_log",
            "git_diff",
            "list_files",
        },
    )
    assert len(candidates) == 1
    assert [step.tool for step in candidates[0].steps] == ["search_files", "read_file"]
    assert "support=3" in candidates[0].steps[0].reason


def test_playbook_miner_prefers_longest_closed_subsequence() -> None:
    # A repeated three-step flow yields one three-step candidate, not the
    # overlapping two-step fragments it contains.
    flow = [
        {"tool": "search_files", "args": {"query": "TODO"}, "status": "executed"},
        {"tool": "read_file", "args": {"path": "README.md"}, "status": "executed"},
        {"tool": "git_status", "args": {}, "status": "executed"},
    ]
    candidates = PlaybookMiner().mine_frequent(
        [list(flow) for _ in range(3)],
        support_threshold=3,
        known_tools={"search_files", "read_file", "git_status"},
    )
    assert len(candidates) == 1
    assert [step.tool for step in candidates[0].steps] == [
        "search_files",
        "read_file",
        "git_status",
    ]


def test_playbook_miner_bounds_candidate_count_and_size() -> None:
    # Many distinct frequent flows: the candidate list is capped.
    sequences = []
    for variant in range(15):
        flow = [
            {"tool": "search_files", "args": {"query": f"q{variant}"}, "status": "executed"},
            {"tool": "read_file", "args": {"path": f"f{variant}.md"}, "status": "executed"},
        ]
        sequences.extend([list(flow) for _ in range(3)])
    candidates = PlaybookMiner().mine_frequent(
        sequences,
        support_threshold=3,
        known_tools={"search_files", "read_file"},
        max_candidates=5,
    )
    assert len(candidates) == 5
    # A very long repeated flow is truncated to the step bound.
    long_flow = [
        {"tool": "read_file", "args": {"path": f"file{i}.md"}, "status": "executed"}
        for i in range(20)
    ]
    long_candidates = PlaybookMiner().mine_frequent(
        [list(long_flow) for _ in range(3)],
        support_threshold=3,
        known_tools={"read_file"},
        max_candidates=1,
    )
    assert len(long_candidates) == 1
    assert len(long_candidates[0].steps) <= 10


async def _seed_conversation_with_message(
    memory: SqliteMemory,
    message: str,
    sequence: list[tuple[str, dict[str, object]]],
) -> str:
    conversation_id = await memory.create_conversation()
    await memory.add_message(conversation_id, "user", message)
    for tool, args in sequence:
        await memory.record_tool_call(
            tool=tool,
            args=args,
            status="executed",
            permission_level=1,
            risk_level="read_only",
            result={"ok": True},
            conversation_id=conversation_id,
        )
    return conversation_id


@pytest.mark.asyncio
async def test_mining_auto_adopts_safe_playbook_with_safe_trigger(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    try:
        sequence = [
            ("search_files", {"query": "TODO"}),
            ("read_file", {"path": "README.md"}),
        ]
        for _ in range(3):
            await _seed_conversation_with_message(memory, "inspect my repo todos", sequence)

        report = await mine_playbook_candidates(
            memory,
            settings_tmp,
            guard=EvolutionWriteGuard(settings_tmp),
            support_threshold=3,
        )
        assert len(report.candidate_ids) == 1
        assert report.adopted_ids == report.candidate_ids
        loaded = PlaybookLoader(settings_tmp.playbooks_path).get(report.candidate_ids[0])
        assert loaded is not None
        assert loaded.status == "active"
        assert loaded.trigger_examples == ["inspect my repo todos"]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_mining_l3_candidate_requires_adoption_approval(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    try:
        # run_command is Level 3: the mined playbook must never auto-adopt.
        sequence = [
            ("search_files", {"query": "TODO"}),
            ("run_command", {"command": "make test"}),
        ]
        for _ in range(3):
            await _seed_conversation_with_message(memory, "run my repo tests", sequence)

        report = await mine_playbook_candidates(
            memory,
            settings_tmp,
            guard=EvolutionWriteGuard(settings_tmp),
            support_threshold=3,
        )
        assert len(report.candidate_ids) == 1
        assert report.adopted_ids == []
        assert report.approval_required_ids == report.candidate_ids
        loaded = PlaybookLoader(settings_tmp.playbooks_path).get(report.candidate_ids[0])
        assert loaded is not None
        assert loaded.status == "candidate"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_mining_without_safe_trigger_stays_candidate(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    try:
        sequence = [
            ("search_files", {"query": "TODO"}),
            ("read_file", {"path": "README.md"}),
        ]
        # Sensitive user messages: no safe trigger can be suggested.
        for _ in range(3):
            await _seed_conversation_with_message(
                memory, "my password is hunter2, check todos", sequence
            )

        report = await mine_playbook_candidates(
            memory,
            settings_tmp,
            guard=EvolutionWriteGuard(settings_tmp),
            support_threshold=3,
        )
        assert len(report.candidate_ids) == 1
        assert report.adopted_ids == []
        loaded = PlaybookLoader(settings_tmp.playbooks_path).get(report.candidate_ids[0])
        assert loaded is not None
        assert loaded.status == "candidate"
        assert loaded.trigger_examples == []
        assert any("no safe trigger" in detail for detail in report.details)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_mining_skips_auto_adopt_on_ambiguous_trigger(settings_tmp) -> None:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    memory = SqliteMemory(database)
    try:
        # An active playbook already owns this trigger phrase.
        loader = PlaybookLoader(settings_tmp.playbooks_path)
        loader.adopt(
            PlaybookDefinition(
                id="existing-playbook",
                name="Existing",
                status="active",
                trigger_examples=["inspect my repo todos"],
                steps=[{"tool": "read_file", "args": {"path": "README.md"}}],
            )
        )
        sequence = [
            ("search_files", {"query": "TODO"}),
            ("read_file", {"path": "README.md"}),
        ]
        for _ in range(3):
            await _seed_conversation_with_message(memory, "inspect my repo todos", sequence)

        report = await mine_playbook_candidates(
            memory,
            settings_tmp,
            guard=EvolutionWriteGuard(settings_tmp),
            support_threshold=3,
        )
        assert len(report.candidate_ids) == 1
        assert report.adopted_ids == []
        loaded = loader.get(report.candidate_ids[0])
        assert loaded is not None
        assert loaded.status == "candidate"
        # The colliding trigger falls back to normal routing behaviour: the
        # loader refuses to match ambiguously once both are present.
        assert any("ambiguous" in detail for detail in report.details)
    finally:
        await database.close()


def test_playbook_api_manual_mine(settings_tmp) -> None:
    import anyio

    async def setup():
        container = await make_container(settings_tmp)
        sequence = [
            ("search_files", {"query": "TODO"}),
            ("read_file", {"path": "README.md"}),
        ]
        for _ in range(3):
            await _seed_tool_sequence(container.memory, sequence)
        return container

    container = anyio.run(setup)
    client = TestClient(create_app(container))
    response = client.post(
        "/playbooks/mine?support_threshold=3&lookback_days=14",
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    assert len(response.json()["mine"]["candidates"]) == 1


def test_playbook_loader_fuzzy_trigger_matches_close_message(settings_tmp) -> None:
    loader = PlaybookLoader(settings_tmp.playbooks_path)
    loader.adopt(_playbook("echo"))
    # Exact containment still matches; a small typo also routes.
    assert loader.match_trigger("please run test playbook now") is not None
    assert loader.match_trigger("run test playbok") is not None
    # Unrelated text never routes.
    assert loader.match_trigger("summarize my README") is None


def test_orchestrator_routes_unambiguous_trigger_to_playbook(settings_tmp) -> None:
    import anyio

    async def scenario() -> None:
        container = await make_container(settings_tmp)
        loader = PlaybookLoader(settings_tmp.playbooks_path)
        loader.adopt(
            PlaybookDefinition(
                id="reminder-playbook",
                name="Reminder playbook",
                agent_id="general_agent",
                status="active",
                trigger_examples=["run my reminder playbook"],
                steps=[{"tool": "create_reminder", "args": {"content": "stand up"}}],
            )
        )
        result = await container.orchestrator.chat("run my reminder playbook")
        assert result.status == "ok"
        assert "reminder-playbook" in result.final_message
        runs = await container.memory.list_playbook_runs(playbook_id="reminder-playbook")
        assert len(runs) == 1
        assert runs[0]["status"] == "completed"
        reminders = await container.memory.list_reminders()
        assert reminders[0].content == "stand up"

    anyio.run(scenario)


def test_orchestrator_ambiguous_trigger_falls_back_to_brain(settings_tmp) -> None:
    import anyio

    async def scenario() -> None:
        container = await make_container(settings_tmp)
        loader = PlaybookLoader(settings_tmp.playbooks_path)
        base = PlaybookDefinition(
            id="one-playbook",
            name="One",
            agent_id="general_agent",
            status="active",
            trigger_examples=["plan my day"],
            steps=[{"tool": "create_reminder", "args": {"content": "one"}}],
        )
        loader.adopt(base)
        loader.adopt(base.model_copy(update={"id": "two-playbook", "name": "Two"}))
        result = await container.orchestrator.chat("plan my day")
        # Ambiguity falls back to normal Brain routing (fake runtime answers).
        assert result.status == "ok"
        assert "playbook" not in result.final_message.casefold()
        assert await container.memory.list_playbook_runs() == []

    anyio.run(scenario)


def test_l3_playbook_adoption_requires_exact_action_approval(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    payload = {
        "id": "danger-playbook",
        "name": "Danger playbook",
        "status": "candidate",
        "agent_id": "coding_agent",
        "trigger_examples": ["apply the danger fix"],
        "steps": [{"tool": "run_command", "args": {"argv": ["pytest"]}}],
    }
    first = client.post("/playbooks/adopt", json=payload, headers=auth(settings_tmp))
    assert first.status_code == 200
    body = first.json()
    assert body["status"] == "pending_approval"
    assert body["required_permission_level"] >= 3
    approval_id = body["approval"]["approval_id"]
    # Not adopted yet: the playbook is not active on disk.
    loader = PlaybookLoader(settings_tmp.playbooks_path)
    assert loader.get("danger-playbook") is None

    # A modified definition must not consume the approval (exact-action bind).
    tampered = {**payload, "steps": [{"tool": "run_command", "args": {"argv": ["ruff"]}}]}
    rejected = client.post(
        f"/playbooks/adopt?approval_id={approval_id}",
        json=tampered,
        headers=auth(settings_tmp),
    )
    assert rejected.status_code in (400, 403)

    accepted = client.post(
        f"/playbooks/adopt?approval_id={approval_id}",
        json=payload,
        headers=auth(settings_tmp),
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "adopted"
    adopted = loader.get("danger-playbook")
    assert adopted is not None
    assert adopted.status == "active"


def test_safe_playbook_adoption_needs_no_approval(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    payload = {
        "id": "safe-playbook",
        "name": "Safe playbook",
        "status": "candidate",
        "agent_id": "general_agent",
        "trigger_examples": ["do the safe thing"],
        "steps": [{"tool": "create_reminder", "args": {"content": "safe"}}],
    }
    response = client.post("/playbooks/adopt", json=payload, headers=auth(settings_tmp))
    assert response.status_code == 200
    assert response.json()["status"] == "adopted"


def test_playbook_api_list_adopt_and_run(settings_tmp) -> None:
    import anyio

    container = anyio.run(make_container, settings_tmp)
    client = TestClient(create_app(container))
    payload = {
        "id": "reminder-playbook",
        "name": "Reminder playbook",
        "status": "candidate",
        "agent_id": "general_agent",
        "trigger_examples": ["remind me"],
        "steps": [{"tool": "create_reminder", "args": {"content": "stand up"}}],
    }
    adopt = client.post("/playbooks/adopt", json=payload, headers=auth(settings_tmp))
    assert adopt.status_code == 200
    listed = client.get("/playbooks", headers=auth(settings_tmp))
    assert listed.status_code == 200
    assert listed.json()["playbooks"][0]["id"] == "reminder-playbook"

    run = client.post(
        "/playbooks/reminder-playbook/run",
        json={},
        headers=auth(settings_tmp),
    )
    assert run.status_code == 200
    assert run.json()["run"]["status"] == "completed"
    reminders = client.get("/reminders", headers=auth(settings_tmp)).json()["reminders"]
    assert reminders[0]["content"] == "stand up"
