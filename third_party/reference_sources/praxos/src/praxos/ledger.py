"""SQLite-backed experience ledger for AI employees."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen

from praxos.models import (
    Account,
    ActionCheck,
    Commitment,
    Customer,
    DecisionRecord,
    Episode,
    Escalation,
    EvidenceReceipt,
    Lesson,
    Outcome,
    Policy,
    ReviewItem,
    Severity,
)


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
STOPWORDS = {
    "about",
    "after",
    "again",
    "alla",
    "and",
    "are",
    "can",
    "che",
    "con",
    "del",
    "della",
    "for",
    "from",
    "gli",
    "have",
    "into",
    "nel",
    "per",
    "say",
    "the",
    "them",
    "they",
    "this",
    "una",
    "uno",
    "will",
    "with",
}
SYNONYMS = {
    "deliver": {"ship", "release", "eta", "date", "delivery"},
    "delivery": {"deliver", "ship", "release", "eta", "date"},
    "ship": {"deliver", "release", "eta", "delivery"},
    "promise": {"commit", "commitment", "guarantee", "assure"},
    "refund": {"reimburse", "credit", "compensation"},
    "contract": {"agreement", "terms", "exception"},
    "escalation": {"incident", "risk", "urgent", "blocker"},
    "roadmap": {"plan", "timeline", "feature", "release"},
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _json_list(value: Iterable[str] | None) -> str:
    return _json(list(value or []))


def _parse_json_list(value: str | None) -> list[str]:
    parsed = _parse_json(value, [])
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _tokens(text: str, *, expand: bool = True) -> set[str]:
    base = {
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if len(token) > 2 and token.lower() not in STOPWORDS
    }
    if not expand:
        return base
    expanded = set(base)
    for token in base:
        expanded.update(SYNONYMS.get(token, set()))
    return expanded


def _trigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[i : i + 3] for i in range(len(compact) - 2)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a.intersection(b)) / len(a.union(b))


def _hybrid_score(query: str, document: str) -> float:
    q = _tokens(query)
    d = _tokens(document)
    token_score = 0.0
    if q and d:
        token_score = len(q.intersection(d)) / math.sqrt(len(q) * len(d))
    char_score = _jaccard(_trigrams(query), _trigrams(document))
    low_query = (query or "").lower()
    low_doc = (document or "").lower()
    phrase_score = 0.0
    if low_query and low_query in low_doc:
        phrase_score = 1.0
    elif low_doc and low_doc in low_query:
        phrase_score = 0.8
    return min(1.0, (0.65 * token_score) + (0.25 * char_score) + (0.10 * phrase_score))


def _score(query: str, document: str) -> float:
    return _hybrid_score(query, document)


def _compact_title(text: str, fallback: str = "Agent lesson") -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return fallback
    return cleaned[:88]


def _first_sentence(text: str, fallback: str) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return fallback
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return parts[0][:240]


def _optional_rerank_scores(query: str, documents: list[str], base_scores: list[float]) -> list[float]:
    """Optionally blend local scores with an external semantic reranker.

    Set PRAXOS_RERANK_URL to an HTTP endpoint accepting:
      {"query": "...", "documents": ["..."]}
    and returning:
      {"scores": [0.0, 0.9, ...]}

    This keeps the default install dependency-free while making semantic
    matching pluggable for teams that already run a reranker.
    """
    url = os.getenv("PRAXOS_RERANK_URL", "").strip()
    if not url or not documents:
        return base_scores
    try:
        payload = _json({"query": query, "documents": documents}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        token = os.getenv("PRAXOS_RERANK_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = Request(url, data=payload, headers=headers, method="POST")
        timeout = float(os.getenv("PRAXOS_RERANK_TIMEOUT", "2.5"))
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw_scores = data.get("scores", [])
        if not isinstance(raw_scores, list) or len(raw_scores) != len(documents):
            return base_scores
        semantic = [max(0.0, min(1.0, float(score))) for score in raw_scores]
        return [
            min(1.0, (0.65 * local) + (0.35 * sem))
            for local, sem in zip(base_scores, semantic)
        ]
    except Exception:
        return base_scores


def _compile_lesson_fields(episode: Episode) -> dict:
    fallback = "Avoid repeating this action without stronger evidence or human approval."
    recommendation = _first_sentence(episode.human_feedback, fallback)
    feedback = (episode.human_feedback or "").strip()
    low = feedback.lower()
    if any(marker in low for marker in ["never", "do not", "don't", "without approval", "without product"]):
        rule = recommendation
    elif feedback:
        rule = f"When handling '{episode.task}', follow this correction: {recommendation}"
    else:
        rule = f"When handling '{episode.task}', review evidence before repeating '{episode.action}'."

    regression_case = (
        f"Given task '{episode.task}', if the agent wants to '{episode.action}', "
        f"expected behavior is: {recommendation}"
    )
    policy_candidate = ""
    if any(marker in low for marker in ["never", "do not", "don't", "must", "without", "approval"]):
        policy_candidate = recommendation
    return {
        "title": _compact_title(episode.task),
        "pattern": f"{episode.task} {episode.action}",
        "rule": rule,
        "recommendation": recommendation,
        "regression_case": regression_case,
        "policy_candidate": policy_candidate,
        "confidence": 0.82 if episode.human_feedback else 0.62,
    }


class ExperienceLedger:
    """Persistent experience ledger.

    Praxos records work episodes, compiles lessons, stores evidence receipts,
    checks future actions, and keeps human review in the loop.
    """

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            env_data_dir = os.getenv("PRAXOS_DATA_DIR")
            data_dir = Path(env_data_dir).expanduser() if env_data_dir else Path.home() / ".praxos"
            db_path = data_dir / "praxos.db"
            if not env_data_dir:
                try:
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                except OSError:
                    db_path = Path.cwd() / ".praxos" / "praxos.db"
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    result TEXT NOT NULL,
                    human_feedback TEXT NOT NULL,
                    source_refs TEXT NOT NULL,
                    evidence_ids TEXT NOT NULL DEFAULT '[]',
                    account_id TEXT NOT NULL DEFAULT '',
                    customer_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_episodes_workspace_created
                    ON episodes(workspace_id, created_at);

                CREATE TABLE IF NOT EXISTS evidence_receipts (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    snippet TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS lessons (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    rule TEXT NOT NULL DEFAULT '',
                    recommendation TEXT NOT NULL,
                    regression_case TEXT NOT NULL DEFAULT '',
                    policy_candidate TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL,
                    evidence_episode_ids TEXT NOT NULL,
                    evidence_receipt_ids TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    use_count INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_lessons_workspace_status
                    ON lessons(workspace_id, status);

                CREATE TABLE IF NOT EXISTS policies (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    instruction TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_policies_workspace_status
                    ON policies(workspace_id, status);

                CREATE TABLE IF NOT EXISTS action_checks (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    action TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reasons TEXT NOT NULL,
                    matched_lesson_ids TEXT NOT NULL,
                    matched_policy_ids TEXT NOT NULL,
                    matched_evidence_ids TEXT NOT NULL DEFAULT '[]',
                    matched_business_ids TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS review_items (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    item_type TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    reviewer TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    external_ref TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS customers (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    external_ref TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS commitments (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS escalations (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    decided_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._ensure_column(conn, "episodes", "evidence_ids", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "episodes", "account_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "episodes", "customer_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "lessons", "rule", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "lessons", "regression_case", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "lessons", "policy_candidate", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "lessons", "evidence_receipt_ids", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "lessons", "review_status", "TEXT NOT NULL DEFAULT 'pending'")
            self._ensure_column(conn, "action_checks", "matched_evidence_ids", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "action_checks", "matched_business_ids", "TEXT NOT NULL DEFAULT '[]'")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def record_episode(
        self,
        *,
        agent_id: str,
        task: str,
        action: str,
        outcome: Outcome = "unknown",
        result: str = "",
        human_feedback: str = "",
        source_refs: Iterable[str] | None = None,
        evidence: Iterable[dict] | None = None,
        account_id: str = "",
        customer_id: str = "",
        workspace_id: str = "default",
        learn: bool = True,
    ) -> Episode:
        if outcome not in {"success", "failure", "blocked", "unknown"}:
            raise ValueError("outcome must be success, failure, blocked, or unknown")
        episode_id = _new_id("ep")
        source_refs_list = list(source_refs or [])
        evidence_ids: list[str] = []
        episode = Episode(
            id=episode_id,
            workspace_id=workspace_id,
            agent_id=agent_id.strip(),
            task=task.strip(),
            action=action.strip(),
            outcome=outcome,
            result=result.strip(),
            human_feedback=human_feedback.strip(),
            source_refs=source_refs_list,
            evidence_ids=evidence_ids,
            account_id=account_id.strip(),
            customer_id=customer_id.strip(),
            created_at=_utcnow(),
        )
        if not episode.agent_id or not episode.task or not episode.action:
            raise ValueError("agent_id, task, and action are required")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes (
                    id, workspace_id, agent_id, task, action, outcome, result,
                    human_feedback, source_refs, evidence_ids, account_id, customer_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.id,
                    episode.workspace_id,
                    episode.agent_id,
                    episode.task,
                    episode.action,
                    episode.outcome,
                    episode.result,
                    episode.human_feedback,
                    _json_list(episode.source_refs),
                    _json_list([]),
                    episode.account_id,
                    episode.customer_id,
                    episode.created_at,
                ),
            )

        default_snippet = episode.human_feedback or episode.result or episode.action
        for source_uri in source_refs_list:
            receipt = self.add_evidence(
                source_uri=source_uri,
                snippet=default_snippet,
                episode_id=episode.id,
                workspace_id=workspace_id,
            )
            evidence_ids.append(receipt.id)

        for item in evidence or []:
            receipt = self.add_evidence(
                source_uri=str(item.get("source_uri") or item.get("source") or ""),
                snippet=str(item.get("snippet") or item.get("text") or default_snippet),
                observed_at=str(item.get("observed_at") or ""),
                confidence=float(item.get("confidence", 0.8)),
                metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                episode_id=episode.id,
                workspace_id=workspace_id,
            )
            evidence_ids.append(receipt.id)

        if evidence_ids:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE episodes SET evidence_ids = ? WHERE id = ?",
                    (_json_list(evidence_ids), episode.id),
                )
            episode = self.get_episode(episode.id) or episode

        if learn and (episode.outcome in {"failure", "blocked"} or episode.human_feedback):
            self.learn_from_episode(episode.id)
        return episode

    def record_outcome(
        self,
        episode_id: str,
        *,
        outcome: Outcome,
        result: str = "",
        human_feedback: str = "",
        learn: bool = True,
    ) -> Episode:
        if outcome not in {"success", "failure", "blocked", "unknown"}:
            raise ValueError("outcome must be success, failure, blocked, or unknown")
        current = self.get_episode(episode_id)
        if current is None:
            raise KeyError(f"episode not found: {episode_id}")

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET outcome = ?, result = ?, human_feedback = ?
                WHERE id = ?
                """,
                (outcome, result.strip(), human_feedback.strip(), episode_id),
            )
        updated = self.get_episode(episode_id)
        if updated is None:
            raise KeyError(f"episode not found after update: {episode_id}")
        if learn and (updated.outcome in {"failure", "blocked"} or updated.human_feedback):
            self.learn_from_episode(updated.id)
        return updated

    def learn_from_feedback(
        self,
        *,
        agent_id: str,
        task: str,
        action: str,
        feedback: str,
        result: str = "",
        source_refs: Iterable[str] | None = None,
        account_id: str = "",
        customer_id: str = "",
        workspace_id: str = "default",
    ) -> Lesson:
        episode = self.record_episode(
            agent_id=agent_id,
            task=task,
            action=action,
            outcome="failure",
            result=result,
            human_feedback=feedback,
            source_refs=source_refs,
            account_id=account_id,
            customer_id=customer_id,
            workspace_id=workspace_id,
            learn=False,
        )
        return self.learn_from_episode(episode.id)

    def add_evidence(
        self,
        *,
        source_uri: str,
        snippet: str,
        episode_id: str = "",
        observed_at: str = "",
        confidence: float = 0.8,
        metadata: dict[str, Any] | None = None,
        workspace_id: str = "default",
    ) -> EvidenceReceipt:
        receipt = EvidenceReceipt(
            id=_new_id("ev"),
            workspace_id=workspace_id,
            episode_id=episode_id,
            source_uri=source_uri.strip(),
            snippet=snippet.strip(),
            observed_at=observed_at.strip() or _utcnow(),
            confidence=max(0.0, min(1.0, float(confidence))),
            metadata=metadata or {},
            created_at=_utcnow(),
        )
        if not receipt.source_uri or not receipt.snippet:
            raise ValueError("source_uri and snippet are required")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evidence_receipts (
                    id, workspace_id, episode_id, source_uri, snippet, observed_at,
                    confidence, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.id,
                    receipt.workspace_id,
                    receipt.episode_id,
                    receipt.source_uri,
                    receipt.snippet,
                    receipt.observed_at,
                    receipt.confidence,
                    _json(receipt.metadata),
                    receipt.created_at,
                ),
            )
        return receipt

    def learn_from_episode(self, episode_id: str) -> Lesson:
        episode = self.get_episode(episode_id)
        if episode is None:
            raise KeyError(f"episode not found: {episode_id}")

        compiled = _compile_lesson_fields(episode)
        similar = self._find_similar_lesson(compiled["pattern"], episode.workspace_id)
        if similar and _score(compiled["pattern"], similar.pattern) >= 0.72:
            evidence = list(dict.fromkeys([*similar.evidence_episode_ids, episode.id]))
            receipt_ids = list(dict.fromkeys([*similar.evidence_receipt_ids, *episode.evidence_ids]))
            confidence = min(0.98, similar.confidence + 0.06)
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE lessons
                    SET evidence_episode_ids = ?, evidence_receipt_ids = ?, confidence = ?
                    WHERE id = ?
                    """,
                    (_json_list(evidence), _json_list(receipt_ids), confidence, similar.id),
                )
            refreshed = self.get_lesson(similar.id)
            if refreshed is None:
                raise RuntimeError("failed to refresh updated lesson")
            return refreshed

        lesson = self.add_lesson(
            title=compiled["title"],
            pattern=compiled["pattern"],
            rule=compiled["rule"],
            recommendation=compiled["recommendation"],
            regression_case=compiled["regression_case"],
            policy_candidate=compiled["policy_candidate"],
            confidence=compiled["confidence"],
            evidence_episode_ids=[episode.id],
            evidence_receipt_ids=episode.evidence_ids,
            workspace_id=episode.workspace_id,
        )
        self._create_review_item(
            workspace_id=episode.workspace_id,
            item_type="lesson",
            item_id=lesson.id,
            summary=f"Review lesson: {lesson.title}",
        )
        return lesson

    def add_lesson(
        self,
        *,
        title: str,
        pattern: str,
        recommendation: str,
        rule: str = "",
        regression_case: str = "",
        policy_candidate: str = "",
        confidence: float = 0.7,
        evidence_episode_ids: Iterable[str] | None = None,
        evidence_receipt_ids: Iterable[str] | None = None,
        workspace_id: str = "default",
        review_status: str = "pending",
    ) -> Lesson:
        lesson = Lesson(
            id=_new_id("les"),
            workspace_id=workspace_id,
            title=title.strip(),
            pattern=pattern.strip(),
            rule=(rule or recommendation).strip(),
            recommendation=recommendation.strip(),
            regression_case=regression_case.strip(),
            policy_candidate=policy_candidate.strip(),
            confidence=max(0.0, min(1.0, float(confidence))),
            evidence_episode_ids=list(evidence_episode_ids or []),
            evidence_receipt_ids=list(evidence_receipt_ids or []),
            status="active",
            review_status=review_status,
            created_at=_utcnow(),
            last_used_at=None,
            use_count=0,
        )
        if not lesson.title or not lesson.pattern or not lesson.recommendation:
            raise ValueError("title, pattern, and recommendation are required")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO lessons (
                    id, workspace_id, title, pattern, rule, recommendation,
                    regression_case, policy_candidate, confidence, evidence_episode_ids,
                    evidence_receipt_ids, status, review_status, created_at, last_used_at, use_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lesson.id,
                    lesson.workspace_id,
                    lesson.title,
                    lesson.pattern,
                    lesson.rule,
                    lesson.recommendation,
                    lesson.regression_case,
                    lesson.policy_candidate,
                    lesson.confidence,
                    _json_list(lesson.evidence_episode_ids),
                    _json_list(lesson.evidence_receipt_ids),
                    lesson.status,
                    lesson.review_status,
                    lesson.created_at,
                    lesson.last_used_at,
                    lesson.use_count,
                ),
            )
        return lesson

    def add_policy(
        self,
        *,
        name: str,
        trigger: str,
        instruction: str,
        severity: Severity = "warn",
        workspace_id: str = "default",
    ) -> Policy:
        if severity not in {"info", "warn", "block"}:
            raise ValueError("severity must be info, warn, or block")
        policy = Policy(
            id=_new_id("pol"),
            workspace_id=workspace_id,
            name=name.strip(),
            trigger=trigger.strip(),
            instruction=instruction.strip(),
            severity=severity,
            status="active",
            created_at=_utcnow(),
        )
        if not policy.name or not policy.trigger or not policy.instruction:
            raise ValueError("name, trigger, and instruction are required")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO policies (
                    id, workspace_id, name, trigger, instruction, severity, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    policy.id,
                    policy.workspace_id,
                    policy.name,
                    policy.trigger,
                    policy.instruction,
                    policy.severity,
                    policy.status,
                    policy.created_at,
                ),
            )
        return policy

    def check_action(
        self,
        *,
        task: str,
        action: str,
        workspace_id: str = "default",
        account_id: str = "",
        lesson_threshold: float = 0.18,
        policy_threshold: float = 0.2,
        business_threshold: float = 0.18,
    ) -> ActionCheck:
        query = f"{task} {action}"
        lesson_matches = self.search_lessons(query, workspace_id=workspace_id, limit=5)
        policy_matches = self.search_policies(query, workspace_id=workspace_id, limit=5)
        business_matches = self.search_business_context(
            query,
            workspace_id=workspace_id,
            account_id=account_id,
            limit=6,
        )

        reasons: list[str] = []
        matched_lesson_ids: list[str] = []
        matched_policy_ids: list[str] = []
        matched_evidence_ids: list[str] = []
        matched_business_ids: list[str] = []
        decision = "allow"
        effective_business_threshold = business_threshold
        if account_id:
            effective_business_threshold = min(business_threshold, 0.075)

        for policy, score in policy_matches:
            if score < policy_threshold:
                continue
            matched_policy_ids.append(policy.id)
            reasons.append(f"Policy matched: {policy.name}. {policy.instruction}")
            if policy.severity == "block":
                decision = "block"
            elif policy.severity == "warn" and decision != "block":
                decision = "warn"

        for lesson, score in lesson_matches:
            if score < lesson_threshold:
                continue
            matched_lesson_ids.append(lesson.id)
            matched_evidence_ids.extend(lesson.evidence_receipt_ids)
            reasons.append(f"Relevant lesson: {lesson.title}. {lesson.recommendation}")
            if decision == "allow":
                decision = "warn"

        for kind, item, score in business_matches:
            if score < effective_business_threshold:
                continue
            item_id = f"{kind}:{item.id}"
            matched_business_ids.append(item_id)
            if kind == "commitment":
                reasons.append(f"Relevant commitment: {item.description}")
            elif kind == "escalation":
                reasons.append(f"Open escalation context: {item.summary}")
            elif kind == "decision":
                reasons.append(f"Relevant decision: {item.decision}")
            if decision == "allow":
                decision = "warn"

        if not reasons:
            reasons.append("No relevant prior experience, policy, or business context matched.")

        check = ActionCheck(
            id=_new_id("chk"),
            workspace_id=workspace_id,
            task=task.strip(),
            action=action.strip(),
            decision=decision,
            reasons=reasons,
            matched_lesson_ids=matched_lesson_ids,
            matched_policy_ids=matched_policy_ids,
            matched_evidence_ids=list(dict.fromkeys(matched_evidence_ids)),
            matched_business_ids=list(dict.fromkeys(matched_business_ids)),
            created_at=_utcnow(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO action_checks (
                    id, workspace_id, task, action, decision, reasons,
                    matched_lesson_ids, matched_policy_ids, matched_evidence_ids,
                    matched_business_ids, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    check.id,
                    check.workspace_id,
                    check.task,
                    check.action,
                    check.decision,
                    _json_list(check.reasons),
                    _json_list(check.matched_lesson_ids),
                    _json_list(check.matched_policy_ids),
                    _json_list(check.matched_evidence_ids),
                    _json_list(check.matched_business_ids),
                    check.created_at,
                ),
            )
        self._mark_lessons_used(matched_lesson_ids)
        return check

    def get_experience(
        self,
        *,
        task: str,
        action: str = "",
        workspace_id: str = "default",
        account_id: str = "",
        limit: int = 5,
    ) -> dict:
        query = f"{task} {action}"
        lesson_matches = self.search_lessons(query, workspace_id=workspace_id, limit=limit)
        lessons = [{"score": score, **lesson.to_dict()} for lesson, score in lesson_matches]
        evidence_ids = []
        for lesson, _ in lesson_matches:
            evidence_ids.extend(lesson.evidence_receipt_ids)
        policies = [
            {"score": score, **policy.to_dict()}
            for policy, score in self.search_policies(query, workspace_id=workspace_id, limit=limit)
        ]
        business = [
            {"kind": kind, "score": score, **item.to_dict()}
            for kind, item, score in self.search_business_context(
                query,
                workspace_id=workspace_id,
                account_id=account_id,
                limit=limit,
            )
        ]
        failures = [
            episode.to_dict()
            for episode in self.list_episodes(
                workspace_id=workspace_id,
                outcome="failure",
                limit=limit,
            )
        ]
        evidence = [receipt.to_dict() for receipt in self.get_evidence_many(evidence_ids)]
        return {
            "query": query.strip(),
            "lessons": lessons,
            "policies": policies,
            "business_context": business,
            "evidence": evidence,
            "recent_failures": failures,
        }

    def search_lessons(
        self,
        query: str,
        *,
        workspace_id: str = "default",
        limit: int = 10,
    ) -> list[tuple[Lesson, float]]:
        lessons = self.list_lessons(workspace_id=workspace_id, limit=500)
        documents = [
            " ".join(
                [
                    lesson.title,
                    lesson.pattern,
                    lesson.rule,
                    lesson.recommendation,
                    lesson.regression_case,
                    lesson.policy_candidate,
                ]
            )
            for lesson in lessons
        ]
        base_scores = [_score(query, doc) for doc in documents]
        final_scores = _optional_rerank_scores(query, documents, base_scores)
        scored = list(zip(lessons, final_scores))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, limit)]

    def search_policies(
        self,
        query: str,
        *,
        workspace_id: str = "default",
        limit: int = 10,
    ) -> list[tuple[Policy, float]]:
        policies = self.list_policies(workspace_id=workspace_id, limit=500)
        low_query = (query or "").lower()
        scored = []
        for policy in policies:
            phrase_score = 0.95 if policy.trigger.lower() in low_query else 0.0
            token_score = _score(query, f"{policy.name} {policy.trigger} {policy.instruction}")
            scored.append((policy, max(phrase_score, token_score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, limit)]

    def create_account(self, *, name: str, external_ref: str = "", workspace_id: str = "default") -> Account:
        account = Account(
            id=_new_id("acct"),
            workspace_id=workspace_id,
            name=name.strip(),
            external_ref=external_ref.strip(),
            created_at=_utcnow(),
        )
        if not account.name:
            raise ValueError("name is required")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO accounts (id, workspace_id, name, external_ref, created_at) VALUES (?, ?, ?, ?, ?)",
                (account.id, account.workspace_id, account.name, account.external_ref, account.created_at),
            )
        return account

    def create_customer(
        self,
        *,
        account_id: str,
        name: str,
        role: str = "",
        external_ref: str = "",
        workspace_id: str = "default",
    ) -> Customer:
        customer = Customer(
            id=_new_id("cust"),
            workspace_id=workspace_id,
            account_id=account_id.strip(),
            name=name.strip(),
            role=role.strip(),
            external_ref=external_ref.strip(),
            created_at=_utcnow(),
        )
        if not customer.account_id or not customer.name:
            raise ValueError("account_id and name are required")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO customers (id, workspace_id, account_id, name, role, external_ref, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    customer.id,
                    customer.workspace_id,
                    customer.account_id,
                    customer.name,
                    customer.role,
                    customer.external_ref,
                    customer.created_at,
                ),
            )
        return customer

    def add_commitment(
        self,
        *,
        account_id: str,
        description: str,
        source_uri: str = "",
        due_at: str = "",
        status: str = "open",
        confidence: float = 0.8,
        workspace_id: str = "default",
    ) -> Commitment:
        item = Commitment(
            id=_new_id("com"),
            workspace_id=workspace_id,
            account_id=account_id.strip(),
            description=description.strip(),
            source_uri=source_uri.strip(),
            due_at=due_at.strip(),
            status=status.strip() or "open",
            confidence=max(0.0, min(1.0, float(confidence))),
            created_at=_utcnow(),
        )
        if not item.account_id or not item.description:
            raise ValueError("account_id and description are required")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO commitments (
                    id, workspace_id, account_id, description, source_uri, due_at,
                    status, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.workspace_id,
                    item.account_id,
                    item.description,
                    item.source_uri,
                    item.due_at,
                    item.status,
                    item.confidence,
                    item.created_at,
                ),
            )
        return item

    def add_escalation(
        self,
        *,
        account_id: str,
        summary: str,
        severity: str = "medium",
        status: str = "open",
        source_uri: str = "",
        workspace_id: str = "default",
    ) -> Escalation:
        item = Escalation(
            id=_new_id("esc"),
            workspace_id=workspace_id,
            account_id=account_id.strip(),
            summary=summary.strip(),
            severity=severity.strip() or "medium",
            status=status.strip() or "open",
            source_uri=source_uri.strip(),
            created_at=_utcnow(),
        )
        if not item.account_id or not item.summary:
            raise ValueError("account_id and summary are required")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO escalations (
                    id, workspace_id, account_id, summary, severity, status, source_uri, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.workspace_id,
                    item.account_id,
                    item.summary,
                    item.severity,
                    item.status,
                    item.source_uri,
                    item.created_at,
                ),
            )
        return item

    def add_decision(
        self,
        *,
        account_id: str,
        decision: str,
        source_uri: str = "",
        decided_at: str = "",
        status: str = "active",
        workspace_id: str = "default",
    ) -> DecisionRecord:
        item = DecisionRecord(
            id=_new_id("dec"),
            workspace_id=workspace_id,
            account_id=account_id.strip(),
            decision=decision.strip(),
            source_uri=source_uri.strip(),
            decided_at=decided_at.strip(),
            status=status.strip() or "active",
            created_at=_utcnow(),
        )
        if not item.account_id or not item.decision:
            raise ValueError("account_id and decision are required")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions (
                    id, workspace_id, account_id, decision, source_uri, decided_at, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.workspace_id,
                    item.account_id,
                    item.decision,
                    item.source_uri,
                    item.decided_at,
                    item.status,
                    item.created_at,
                ),
            )
        return item

    def search_business_context(
        self,
        query: str,
        *,
        workspace_id: str = "default",
        account_id: str = "",
        limit: int = 10,
    ) -> list[tuple[str, Commitment | Escalation | DecisionRecord, float]]:
        scored: list[tuple[str, Commitment | Escalation | DecisionRecord, float]] = []
        for item in self.list_commitments(workspace_id=workspace_id, account_id=account_id, limit=500):
            scored.append(("commitment", item, _score(query, item.description)))
        for item in self.list_escalations(workspace_id=workspace_id, account_id=account_id, limit=500):
            scored.append(("escalation", item, _score(query, item.summary)))
        for item in self.list_decisions(workspace_id=workspace_id, account_id=account_id, limit=500):
            scored.append(("decision", item, _score(query, item.decision)))
        scored.sort(key=lambda item: item[2], reverse=True)
        return scored[: max(1, limit)]

    def list_accounts(self, *, workspace_id: str = "default", limit: int = 100) -> list[Account]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM accounts WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ?",
                (workspace_id, max(1, limit)),
            ).fetchall()
        return [self._row_to_account(row) for row in rows]

    def list_commitments(
        self,
        *,
        workspace_id: str = "default",
        account_id: str = "",
        limit: int = 100,
    ) -> list[Commitment]:
        sql = "SELECT * FROM commitments WHERE workspace_id = ?"
        params: list[object] = [workspace_id]
        if account_id:
            sql += " AND account_id = ?"
            params.append(account_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_commitment(row) for row in rows]

    def list_escalations(
        self,
        *,
        workspace_id: str = "default",
        account_id: str = "",
        limit: int = 100,
    ) -> list[Escalation]:
        sql = "SELECT * FROM escalations WHERE workspace_id = ?"
        params: list[object] = [workspace_id]
        if account_id:
            sql += " AND account_id = ?"
            params.append(account_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_escalation(row) for row in rows]

    def list_decisions(
        self,
        *,
        workspace_id: str = "default",
        account_id: str = "",
        limit: int = 100,
    ) -> list[DecisionRecord]:
        sql = "SELECT * FROM decisions WHERE workspace_id = ?"
        params: list[object] = [workspace_id]
        if account_id:
            sql += " AND account_id = ?"
            params.append(account_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_decision(row) for row in rows]

    def get_episode(self, episode_id: str) -> Episode | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        return self._row_to_episode(row) if row else None

    def get_lesson(self, lesson_id: str) -> Lesson | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
        return self._row_to_lesson(row) if row else None

    def get_evidence_many(self, evidence_ids: Iterable[str]) -> list[EvidenceReceipt]:
        ids = list(dict.fromkeys(evidence_ids))
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM evidence_receipts WHERE id IN ({marks})", ids).fetchall()
        by_id = {row["id"]: self._row_to_evidence(row) for row in rows}
        return [by_id[item_id] for item_id in ids if item_id in by_id]

    def list_episodes(
        self,
        *,
        workspace_id: str = "default",
        outcome: Outcome | None = None,
        limit: int = 20,
    ) -> list[Episode]:
        sql = "SELECT * FROM episodes WHERE workspace_id = ?"
        params: list[object] = [workspace_id]
        if outcome:
            sql += " AND outcome = ?"
            params.append(outcome)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def list_lessons(self, *, workspace_id: str = "default", limit: int = 20) -> list[Lesson]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM lessons
                WHERE workspace_id = ? AND status = 'active'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (workspace_id, max(1, limit)),
            ).fetchall()
        return [self._row_to_lesson(row) for row in rows]

    def list_policies(self, *, workspace_id: str = "default", limit: int = 20) -> list[Policy]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM policies
                WHERE workspace_id = ? AND status = 'active'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (workspace_id, max(1, limit)),
            ).fetchall()
        return [self._row_to_policy(row) for row in rows]

    def list_review_items(
        self,
        *,
        workspace_id: str = "default",
        status: str = "pending",
        limit: int = 50,
    ) -> list[ReviewItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM review_items
                WHERE workspace_id = ? AND status = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (workspace_id, status, max(1, limit)),
            ).fetchall()
        return [self._row_to_review_item(row) for row in rows]

    def review_item(self, item_id: str, *, approve: bool, reviewer: str = "human") -> ReviewItem:
        status = "approved" if approve else "rejected"
        now = _utcnow()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM review_items WHERE id = ?", (item_id,)).fetchone()
            if not row:
                raise KeyError(f"review item not found: {item_id}")
            item = self._row_to_review_item(row)
            conn.execute(
                "UPDATE review_items SET status = ?, reviewed_at = ?, reviewer = ? WHERE id = ?",
                (status, now, reviewer, item_id),
            )
            if item.item_type == "lesson":
                lesson_status = "approved" if approve else "rejected"
                active_status = "active" if approve else "archived"
                conn.execute(
                    "UPDATE lessons SET review_status = ?, status = ? WHERE id = ?",
                    (lesson_status, active_status, item.item_id),
                )
        return self.get_review_item(item_id)

    def get_review_item(self, item_id: str) -> ReviewItem:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM review_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            raise KeyError(f"review item not found: {item_id}")
        return self._row_to_review_item(row)

    def stats(self, *, workspace_id: str = "default") -> dict:
        with self._connect() as conn:
            episode_count = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()[0]
            lesson_count = conn.execute(
                "SELECT COUNT(*) FROM lessons WHERE workspace_id = ? AND status = 'active'",
                (workspace_id,),
            ).fetchone()[0]
            policy_count = conn.execute(
                "SELECT COUNT(*) FROM policies WHERE workspace_id = ? AND status = 'active'",
                (workspace_id,),
            ).fetchone()[0]
            check_count = conn.execute(
                "SELECT COUNT(*) FROM action_checks WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()[0]
            evidence_count = conn.execute(
                "SELECT COUNT(*) FROM evidence_receipts WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()[0]
            review_count = conn.execute(
                "SELECT COUNT(*) FROM review_items WHERE workspace_id = ? AND status = 'pending'",
                (workspace_id,),
            ).fetchone()[0]
            account_count = conn.execute(
                "SELECT COUNT(*) FROM accounts WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()[0]
        return {
            "workspace_id": workspace_id,
            "episodes": int(episode_count),
            "lessons": int(lesson_count),
            "policies": int(policy_count),
            "action_checks": int(check_count),
            "evidence_receipts": int(evidence_count),
            "pending_reviews": int(review_count),
            "accounts": int(account_count),
            "db_path": str(self.db_path),
        }

    def _create_review_item(self, *, workspace_id: str, item_type: str, item_id: str, summary: str) -> ReviewItem:
        item = ReviewItem(
            id=_new_id("rev"),
            workspace_id=workspace_id,
            item_type=item_type,
            item_id=item_id,
            status="pending",
            summary=summary,
            created_at=_utcnow(),
            reviewed_at=None,
            reviewer="",
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO review_items (
                    id, workspace_id, item_type, item_id, status, summary, created_at, reviewed_at, reviewer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.workspace_id,
                    item.item_type,
                    item.item_id,
                    item.status,
                    item.summary,
                    item.created_at,
                    item.reviewed_at,
                    item.reviewer,
                ),
            )
        return item

    def _find_similar_lesson(self, pattern: str, workspace_id: str) -> Lesson | None:
        matches = self.search_lessons(pattern, workspace_id=workspace_id, limit=1)
        if not matches:
            return None
        lesson, score = matches[0]
        return lesson if score > 0 else None

    def _mark_lessons_used(self, lesson_ids: Iterable[str]) -> None:
        ids = list(lesson_ids)
        if not ids:
            return
        now = _utcnow()
        with self._connect() as conn:
            conn.executemany(
                """
                UPDATE lessons
                SET last_used_at = ?, use_count = use_count + 1
                WHERE id = ?
                """,
                [(now, lesson_id) for lesson_id in ids],
            )

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> Episode:
        return Episode(
            id=row["id"],
            workspace_id=row["workspace_id"],
            agent_id=row["agent_id"],
            task=row["task"],
            action=row["action"],
            outcome=row["outcome"],
            result=row["result"],
            human_feedback=row["human_feedback"],
            source_refs=_parse_json_list(row["source_refs"]),
            evidence_ids=_parse_json_list(row["evidence_ids"]),
            account_id=row["account_id"],
            customer_id=row["customer_id"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_evidence(row: sqlite3.Row) -> EvidenceReceipt:
        metadata = _parse_json(row["metadata"], {})
        if not isinstance(metadata, dict):
            metadata = {}
        return EvidenceReceipt(
            id=row["id"],
            workspace_id=row["workspace_id"],
            episode_id=row["episode_id"],
            source_uri=row["source_uri"],
            snippet=row["snippet"],
            observed_at=row["observed_at"],
            confidence=float(row["confidence"]),
            metadata=metadata,
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_lesson(row: sqlite3.Row) -> Lesson:
        return Lesson(
            id=row["id"],
            workspace_id=row["workspace_id"],
            title=row["title"],
            pattern=row["pattern"],
            rule=row["rule"],
            recommendation=row["recommendation"],
            regression_case=row["regression_case"],
            policy_candidate=row["policy_candidate"],
            confidence=float(row["confidence"]),
            evidence_episode_ids=_parse_json_list(row["evidence_episode_ids"]),
            evidence_receipt_ids=_parse_json_list(row["evidence_receipt_ids"]),
            status=row["status"],
            review_status=row["review_status"],
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
            use_count=int(row["use_count"]),
        )

    @staticmethod
    def _row_to_policy(row: sqlite3.Row) -> Policy:
        return Policy(
            id=row["id"],
            workspace_id=row["workspace_id"],
            name=row["name"],
            trigger=row["trigger"],
            instruction=row["instruction"],
            severity=row["severity"],
            status=row["status"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_review_item(row: sqlite3.Row) -> ReviewItem:
        return ReviewItem(
            id=row["id"],
            workspace_id=row["workspace_id"],
            item_type=row["item_type"],
            item_id=row["item_id"],
            status=row["status"],
            summary=row["summary"],
            created_at=row["created_at"],
            reviewed_at=row["reviewed_at"],
            reviewer=row["reviewer"],
        )

    @staticmethod
    def _row_to_account(row: sqlite3.Row) -> Account:
        return Account(
            id=row["id"],
            workspace_id=row["workspace_id"],
            name=row["name"],
            external_ref=row["external_ref"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_commitment(row: sqlite3.Row) -> Commitment:
        return Commitment(
            id=row["id"],
            workspace_id=row["workspace_id"],
            account_id=row["account_id"],
            description=row["description"],
            source_uri=row["source_uri"],
            due_at=row["due_at"],
            status=row["status"],
            confidence=float(row["confidence"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_escalation(row: sqlite3.Row) -> Escalation:
        return Escalation(
            id=row["id"],
            workspace_id=row["workspace_id"],
            account_id=row["account_id"],
            summary=row["summary"],
            severity=row["severity"],
            status=row["status"],
            source_uri=row["source_uri"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_decision(row: sqlite3.Row) -> DecisionRecord:
        return DecisionRecord(
            id=row["id"],
            workspace_id=row["workspace_id"],
            account_id=row["account_id"],
            decision=row["decision"],
            source_uri=row["source_uri"],
            decided_at=row["decided_at"],
            status=row["status"],
            created_at=row["created_at"],
        )
