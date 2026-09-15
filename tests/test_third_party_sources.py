from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from apps.runner.third_party_sources import (
    ThirdPartyManifest,
    ThirdPartySource,
    ThirdPartySourceError,
    build_colibri,
    inspect_source_manifest,
    load_source_manifest,
    manifest_identity_digest,
)
from scripts.check_source_hygiene import forbidden_reason

ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_manifest_is_explicitly_metadata_only_until_source_is_reviewed() -> None:
    manifest = load_source_manifest(ROOT)
    assert manifest.colibri.status == "metadata_only"
    assert len(manifest.entries) == 9
    result = inspect_source_manifest(ROOT)
    assert result["manifest_valid"] is True
    assert result["ready"] is False
    colibri = next(entry for entry in result["entries"] if entry["id"] == "colibri")
    assert colibri["reason"] == "source_not_staged"
    assert colibri["production_dependency"] is False
    assert colibri["license_notice_present"] is False
    assert "/" not in str(result.get("root", ""))


def test_manifest_digest_is_stable_and_does_not_hash_source_contents() -> None:
    first = manifest_identity_digest(ROOT)
    second = manifest_identity_digest(ROOT)
    assert first == second
    assert len(first) == 64


def test_third_party_sources_are_not_python_production_packages() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packages = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "third_party" not in packages
    assert all(entry.kind == "reference" for entry in load_source_manifest(ROOT).entries[1:])


def test_manifest_rejects_paths_outside_third_party(tmp_path: Path) -> None:
    payload = {
        "manifest_id": "test",
        "schema_version": "1",
        "entries": [
            {
                "id": "colibri",
                "kind": "runtime",
                "status": "metadata_only",
                "upstream": {"name": "Colibri", "revision": None},
                "source_path": "third_party/../outside",
                "license_files": [],
                "notice_files": [],
                "adaptation_metadata": None,
                "production_dependency": False,
                "fingerprint_scope": "metadata",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ThirdPartySourceError, match="outside"):
        load_source_manifest(tmp_path, path)


def test_vendored_entry_requires_source_pin_and_preserved_notices(tmp_path: Path) -> None:
    source = tmp_path / "third_party" / "colibri" / "source"
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text("project(colibri)\n", encoding="utf-8")
    (tmp_path / "third_party" / "colibri" / "LICENSE").write_text("license", encoding="utf-8")
    (tmp_path / "third_party" / "colibri" / "NOTICE").write_text("notice", encoding="utf-8")
    adaptation = tmp_path / "third_party" / "colibri" / "APRIL_ADAPTATION.md"
    adaptation.write_text("adaptation", encoding="utf-8")
    payload = {
        "manifest_id": "test",
        "schema_version": "1",
        "entries": [
            {
                "id": "colibri",
                "kind": "runtime",
                "status": "vendored",
                "upstream": {"name": "Colibri", "revision": "a" * 40},
                "source_path": "third_party/colibri/source",
                "license_files": ["third_party/colibri/LICENSE"],
                "notice_files": ["third_party/colibri/NOTICE"],
                "adaptation_metadata": "third_party/colibri/APRIL_ADAPTATION.md",
                "production_dependency": False,
                "fingerprint_scope": "source metadata",
            }
        ],
    }
    manifest = tmp_path / "third_party" / "source-manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    result = inspect_source_manifest(tmp_path, manifest)
    assert result["ready"] is True
    assert result["entries"][0]["revision_pinned"] is True
    assert result["entries"][0]["license_notice_present"] is True


def test_build_colibri_does_not_proceed_without_reviewed_source() -> None:
    with pytest.raises(ThirdPartySourceError, match="not staged"):
        build_colibri(ROOT)


def test_source_hygiene_rejects_tracked_third_party_build_and_model_artifacts() -> None:
    assert forbidden_reason("third_party/colibri/build/log.txt")
    assert forbidden_reason("third_party/colibri/models/model.gguf")
    assert forbidden_reason("third_party/reference_sources/foo/weights.bin")
    assert forbidden_reason("third_party/colibri/source/CMakeLists.txt") is None
    assert forbidden_reason("third_party/colibri/APRIL_ADAPTATION.md") is None


def _entry(entry_id: str = "colibri", **updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": entry_id,
        "kind": "runtime",
        "status": "metadata_only",
        "upstream": {"name": "Colibri", "revision": None},
        "source_path": "third_party/colibri/source",
        "license_files": [],
        "notice_files": [],
        "adaptation_metadata": None,
        "production_dependency": False,
        "fingerprint_scope": "metadata",
    }
    value.update(updates)
    return value


def _manifest(tmp_path: Path, *entries: dict[str, object]) -> Path:
    target = tmp_path / "third_party" / "source-manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"manifest_id": "test", "schema_version": "1", "entries": list(entries)}),
        encoding="utf-8",
    )
    return target


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("[1]", "must be an object"),
        (json.dumps({"manifest_id": "x", "schema_version": "1", "entries": [1]}), "invalid entry"),
        (json.dumps({"manifest_id": "x", "schema_version": "1", "entries": []}), "no entries"),
        (json.dumps({"manifest_id": "x", "entries": [_entry()]}), "schema version"),
        (json.dumps({"schema_version": "1", "entries": [_entry()]}), "manifest ID"),
    ],
)
def test_manifest_rejects_malformed_top_level_documents(
    tmp_path: Path, raw: str, message: str
) -> None:
    path = tmp_path / "third_party" / "source-manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ThirdPartySourceError, match=message):
        load_source_manifest(tmp_path)


def test_manifest_rejects_bad_entry_metadata(tmp_path: Path) -> None:
    cases = [
        (_entry("other"), "must define Colibri"),
        (_entry(upstream={"name": "", "revision": None}), "upstream name"),
        (_entry(upstream={"name": "Colibri", "repository": 4}), "repository URL"),
        (_entry(upstream={"name": "Colibri", "revision": "tag"}), "full Git SHA"),
        (_entry(license_files="bad"), "license or notice"),
        (_entry(source_path="third_party/../outside"), "outside"),
        (_entry(source_path="outside"), "under third_party"),
        (_entry(adaptation_metadata=""), "invalid relative path"),
        (_entry(production_dependency="yes"), "dependency flag"),
        (_entry(fingerprint_scope=""), "fingerprint scope"),
        (_entry(kind="reference", production_dependency=True), "production dependency"),
    ]
    for entry, message in cases:
        path = _manifest(tmp_path, entry)
        with pytest.raises(ThirdPartySourceError, match=message):
            load_source_manifest(tmp_path, path)


def test_manifest_rejects_duplicate_and_digest_read_errors(tmp_path: Path) -> None:
    path = _manifest(tmp_path, _entry(), _entry())
    with pytest.raises(ThirdPartySourceError, match="Duplicate"):
        load_source_manifest(tmp_path, path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ThirdPartySourceError, match="unreadable"):
        manifest_identity_digest(tmp_path, path)
    path.unlink()
    with pytest.raises(ThirdPartySourceError, match="unreadable"):
        manifest_identity_digest(tmp_path, path)


def test_manifest_colibri_property_fails_when_constructed_without_colibri() -> None:
    entry = ThirdPartySource(
        id="other",
        kind="reference",
        status="metadata_only",
        upstream_name="other",
        upstream_repository=None,
        revision=None,
        source_path=Path("third_party/reference_sources/other"),
        license_files=(),
        notice_files=(),
        adaptation_metadata=None,
        production_dependency=False,
        fingerprint_scope="metadata",
    )
    with pytest.raises(ThirdPartySourceError, match="Colibri"):
        _ = ThirdPartyManifest("1", "test", (entry,)).colibri


def test_doctor_reports_invalid_manifest_without_raising(tmp_path: Path) -> None:
    path = tmp_path / "third_party" / "source-manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")
    result = inspect_source_manifest(tmp_path, path)
    assert result == {
        "ready": False,
        "manifest_valid": False,
        "reason": "Third-party source manifest is unreadable.",
    }


def test_build_colibri_validates_parallelism_and_build_entrypoint(tmp_path: Path) -> None:
    with pytest.raises(ThirdPartySourceError, match="parallelism"):
        build_colibri(ROOT, jobs=0)
    source = tmp_path / "third_party" / "colibri" / "source"
    source.mkdir(parents=True)
    (tmp_path / "third_party" / "colibri" / "LICENSE").write_text("L", encoding="utf-8")
    (tmp_path / "third_party" / "colibri" / "NOTICE").write_text("N", encoding="utf-8")
    manifest = _manifest(
        tmp_path,
        _entry(
            status="vendored",
            upstream={"name": "Colibri", "revision": "b" * 40},
            license_files=["third_party/colibri/LICENSE"],
            notice_files=["third_party/colibri/NOTICE"],
        ),
    )
    with pytest.raises(ThirdPartySourceError, match="CMakeLists"):
        build_colibri(tmp_path, path=manifest)


def test_build_colibri_runs_only_the_fixed_local_commands(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "third_party" / "colibri" / "source"
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text("project(colibri)\n", encoding="utf-8")
    (tmp_path / "third_party" / "colibri" / "LICENSE").write_text("L", encoding="utf-8")
    (tmp_path / "third_party" / "colibri" / "NOTICE").write_text("N", encoding="utf-8")
    manifest = _manifest(
        tmp_path,
        _entry(
            status="vendored",
            upstream={"name": "Colibri", "revision": "c" * 40},
            license_files=["third_party/colibri/LICENSE"],
            notice_files=["third_party/colibri/NOTICE"],
        ),
    )
    calls: list[tuple[list[str], Path]] = []
    monkeypatch.setattr("apps.runner.third_party_sources.shutil.which", lambda _: "cmake")
    monkeypatch.setattr(
        "apps.runner.third_party_sources.subprocess.run",
        lambda argv, check, cwd: calls.append((argv, cwd)),
    )
    result = build_colibri(tmp_path, jobs=3, path=manifest)
    assert result["built"] is True
    assert len(calls) == 2
    assert calls[0][0][0] == "cmake"
    assert calls[0][0][-1] == "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    assert calls[1][0][1:3] == ["--build", str(tmp_path / "third_party/colibri/build")]
    assert all(cwd == tmp_path for _, cwd in calls)


def test_build_colibri_reports_missing_cmake(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "third_party" / "colibri" / "source"
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text("project(colibri)\n", encoding="utf-8")
    (tmp_path / "third_party" / "colibri" / "LICENSE").write_text("L", encoding="utf-8")
    (tmp_path / "third_party" / "colibri" / "NOTICE").write_text("N", encoding="utf-8")
    manifest = _manifest(
        tmp_path,
        _entry(
            status="vendored",
            upstream={"name": "Colibri", "revision": "d" * 40},
            license_files=["third_party/colibri/LICENSE"],
            notice_files=["third_party/colibri/NOTICE"],
        ),
    )
    monkeypatch.setattr("apps.runner.third_party_sources.shutil.which", lambda _: None)
    with pytest.raises(ThirdPartySourceError, match="CMake"):
        build_colibri(tmp_path, path=manifest)
