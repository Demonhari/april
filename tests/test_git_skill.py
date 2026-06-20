from __future__ import annotations

import subprocess

import pytest

from skills.git.git_branch import git_branch
from skills.git.git_status import git_status


@pytest.mark.asyncio
async def test_git_read_only_skills(settings_tmp) -> None:
    subprocess.run(["git", "init"], cwd=settings_tmp.home, check=True, stdout=subprocess.PIPE)
    result = await git_status({"repo_path": str(settings_tmp.home)})
    assert result.ok
    branch = await git_branch({"repo_path": str(settings_tmp.home)})
    assert branch.ok
