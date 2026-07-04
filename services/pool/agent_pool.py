from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from april_common.time import utc_now
from services.memory.sqlite_memory import SqliteMemory

# Display call signs for the named agent pool. These are presentation
# metadata only: permissions, tools, and routing always key off the agent id.
CALL_SIGNS: dict[str, str] = {
    "general_agent": "Prime",
    "reasoning_agent": "Sage",
    "creative_agent": "Muse",
    "reading_agent": "Scout",
    "memory_agent": "Archive",
    "coding_agent": "Forge",
    "system_action_agent": "Hand",
}

ROLLING_WINDOW_DAYS = 30


@dataclass(frozen=True, slots=True)
class AgentScorecard:
    """Honest per-agent counters straight from agent_runs/feedback_events.

    There is deliberately no synthesized "score" figure: every field is a
    count or timestamp persisted by real runs and real user feedback.
    """

    agent: str
    call_sign: str
    total_runs: int
    recent_runs: int
    ok_runs: int
    error_runs: int
    pending_runs: int
    feedback_good: int
    feedback_bad: int
    last_run_at: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "call_sign": self.call_sign,
            "total_runs": self.total_runs,
            "recent_runs": self.recent_runs,
            "rolling_window_days": ROLLING_WINDOW_DAYS,
            "ok_runs": self.ok_runs,
            "error_runs": self.error_runs,
            "pending_runs": self.pending_runs,
            "feedback_good": self.feedback_good,
            "feedback_bad": self.feedback_bad,
            "last_run_at": self.last_run_at,
        }


class AgentPool:
    """Read-only scorecards for the named specialist pool.

    The pool never executes anything and never bypasses Runtime or the
    permission engine — it only aggregates persisted run and feedback rows.
    """

    def __init__(self, memory: SqliteMemory, *, known_agents: list[str] | None = None) -> None:
        self.memory = memory
        self.known_agents = known_agents if known_agents is not None else list(CALL_SIGNS)

    async def scorecards(self) -> list[AgentScorecard]:
        window_start = (
            (utc_now() - timedelta(days=ROLLING_WINDOW_DAYS))
            .isoformat()
            .replace("+00:00", "Z")
        )
        run_rows = await self.memory.database.fetchall(
            """
            SELECT
                agent,
                COUNT(*) AS total_runs,
                SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS recent_runs,
                SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok_runs,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_runs,
                SUM(CASE WHEN status = 'pending_approval' THEN 1 ELSE 0 END)
                    AS pending_runs,
                MAX(created_at) AS last_run_at
            FROM agent_runs
            GROUP BY agent
            """,
            (window_start,),
        )
        feedback_rows = await self.memory.database.fetchall(
            """
            SELECT
                runs.agent AS agent,
                SUM(CASE WHEN events.rating = 'good' THEN 1 ELSE 0 END) AS good,
                SUM(CASE WHEN events.rating = 'bad' THEN 1 ELSE 0 END) AS bad
            FROM feedback_events AS events
            JOIN agent_runs AS runs ON runs.id = events.agent_run_id
            GROUP BY runs.agent
            """
        )
        runs_by_agent = {str(row["agent"]): row for row in run_rows}
        feedback_by_agent = {str(row["agent"]): row for row in feedback_rows}
        agents = list(
            dict.fromkeys([*self.known_agents, *runs_by_agent, *feedback_by_agent])
        )
        cards: list[AgentScorecard] = []
        for agent in agents:
            run = runs_by_agent.get(agent)
            feedback = feedback_by_agent.get(agent)
            cards.append(
                AgentScorecard(
                    agent=agent,
                    call_sign=CALL_SIGNS.get(agent, agent),
                    total_runs=int(run["total_runs"]) if run else 0,
                    recent_runs=int(run["recent_runs"] or 0) if run else 0,
                    ok_runs=int(run["ok_runs"] or 0) if run else 0,
                    error_runs=int(run["error_runs"] or 0) if run else 0,
                    pending_runs=int(run["pending_runs"] or 0) if run else 0,
                    feedback_good=int(feedback["good"] or 0) if feedback else 0,
                    feedback_bad=int(feedback["bad"] or 0) if feedback else 0,
                    last_run_at=str(run["last_run_at"]) if run and run["last_run_at"] else None,
                )
            )
        return cards
