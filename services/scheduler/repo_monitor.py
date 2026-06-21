from __future__ import annotations

from dataclasses import dataclass

from april_common.time import utc_now_iso
from services.memory.sqlite_memory import SqliteMemory
from skills.git.common import run_git


@dataclass(slots=True)
class RepoActivity:
    project_name: str
    project_path: str
    head_sha: str | None
    dirty_count: int
    new_commits: bool


async def compute_repo_activity(memory: SqliteMemory, *, persist: bool) -> list[RepoActivity]:
    """Read-only git scan of every registered project.

    For each project, reads HEAD and `status --porcelain` through the validated
    run_git helper (argv array, shell=False). Projects that are not git repos,
    whose path is gone, or whose git commands fail are skipped silently.

    When persist is True the per-project baseline snapshot is advanced so the
    next briefing can detect new commits. When persist is False (preview) the
    baseline is left untouched, so previews are idempotent.
    """
    activities: list[RepoActivity] = []
    for project in await memory.list_projects():
        try:
            rc1, head_out, _ = await run_git(project.path, ["rev-parse", "HEAD"])
            rc2, status_out, _ = await run_git(project.path, ["status", "--porcelain"])
        except Exception:
            # Not a git repo, path gone, or command unavailable: skip, never raise.
            continue
        if rc1 != 0 or rc2 != 0:
            continue

        head_sha = head_out.strip() or None
        dirty_count = sum(1 for line in status_out.splitlines() if line.strip())

        prior = await memory.get_repo_snapshot(project.id)
        new_commits = bool(
            prior and prior["head_sha"] and head_sha and prior["head_sha"] != head_sha
        )

        if persist:
            await memory.upsert_repo_snapshot(project.id, head_sha, dirty_count, utc_now_iso())

        activities.append(
            RepoActivity(
                project_name=project.name,
                project_path=project.path,
                head_sha=head_sha,
                dirty_count=dirty_count,
                new_commits=new_commits,
            )
        )
    return activities
