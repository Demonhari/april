from __future__ import annotations

import json
from pathlib import Path

import yaml

from skills.playbooks.schema import PlaybookDefinition


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
        normalized = text.casefold()
        matches = [
            playbook
            for playbook in self.list()
            if playbook.status == "active"
            and any(example.casefold() in normalized for example in playbook.trigger_examples)
        ]
        if len(matches) == 1:
            return matches[0]
        return None
