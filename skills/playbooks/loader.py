from __future__ import annotations

import json
from pathlib import Path

import yaml

from april_common.text import normalized_edit_distance
from skills.playbooks.schema import PlaybookDefinition

# A whole-message trigger match tolerates this much normalized edit distance,
# so "run test playbok" still routes while unrelated text never does.
FUZZY_TRIGGER_MAX_DISTANCE = 0.2


class PlaybookLoader:
    def __init__(self, root: Path) -> None:
        self.root = root

    def list(self) -> list[PlaybookDefinition]:
        self.root.mkdir(parents=True, exist_ok=True)
        playbooks: list[PlaybookDefinition] = []
        for path in sorted(self.root.iterdir()):
            if path.suffix not in {".json", ".yaml", ".yml"} or not path.is_file():
                continue
            playbooks.append(self.load_path(path))
        return playbooks

    def get(self, playbook_id: str) -> PlaybookDefinition | None:
        for suffix in (".yaml", ".yml", ".json"):
            path = self.root / f"{playbook_id}{suffix}"
            if path.exists() and path.is_file():
                return self.load_path(path)
        return None

    def load_path(self, path: Path) -> PlaybookDefinition:
        resolved = path.expanduser().resolve()
        if resolved.parent != self.root.expanduser().resolve():
            raise RuntimeError("playbooks must be loaded from the configured playbooks directory")
        if resolved.suffix == ".json":
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        else:
            payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        return PlaybookDefinition.model_validate(payload)

    def adopt(self, playbook: PlaybookDefinition) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        active = playbook.model_copy(update={"status": "active"})
        target = (self.root / f"{active.id}.yaml").resolve()
        if target.parent != self.root.resolve():
            raise RuntimeError("playbook target escaped the configured playbooks directory")
        target.write_text(yaml.safe_dump(active.model_dump(), sort_keys=True), encoding="utf-8")
        return target

    def match_trigger(self, text: str) -> PlaybookDefinition | None:
        """Route to a playbook only on an unambiguous trigger match.

        A trigger example matches when it is contained in the message (exact)
        or the whole message is within a small edit distance of the example
        (fuzzy). More than one matching playbook is ambiguous and returns
        ``None`` so the caller falls back to normal Brain routing.
        """
        normalized = " ".join(text.casefold().split())
        if not normalized:
            return None
        matches = [
            playbook
            for playbook in self.list()
            if playbook.status == "active"
            and any(
                self._example_matches(example, normalized) for example in playbook.trigger_examples
            )
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    @staticmethod
    def _example_matches(example: str, normalized_message: str) -> bool:
        normalized_example = " ".join(example.casefold().split())
        if not normalized_example:
            return False
        if normalized_example in normalized_message:
            return True
        return (
            normalized_edit_distance(normalized_example, normalized_message)
            <= FUZZY_TRIGGER_MAX_DISTANCE
        )
