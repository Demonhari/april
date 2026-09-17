from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

import pytest

from apps.runner.third_party_sources import (
    ThirdPartyManifest,
    ThirdPartySource,
    ThirdPartySourceError,
    _run_local_build,
    _safe_is_dir,
    _safe_is_file,
    _safe_relative_path,
    _vendor_tree_details,
    build_colibri,
    inspect_source_manifest,
    load_source_manifest,
    manifest_identity_digest,
    resolve_colibri_python,
    run_colibri_command,
    vendor_snapshot_digest,
)
from scripts.check_source_hygiene import forbidden_reason

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = {
    "colibri",
    "mnem",
    "agentmw",
    "anybridge",
    "tools-factory",
    "praxos",
    "openvurp",
    "llm-use",
    "coocon",
}


def test_checked_in_manifest_contains_all_vendored_sources() -> None:
    manifest = load_source_manifest(ROOT)
    assert {entry.id for entry in manifest.entries} == EXPECTED_IDS
    assert all(entry.status == "vendored" for entry in manifest.entries)
    assert manifest.provenance_status == "complete"
    result = inspect_source_manifest(ROOT)
    assert result["ready"] is True
    assert all(entry["ready"] for entry in result["entries"])
    assert all(entry["snapshot_matches"] for entry in result["entries"])


def test_vendored_paths_licenses_notices_and_adaptations_exist() -> None:
    manifest = load_source_manifest(ROOT)
    for entry in manifest.entries:
        assert (ROOT / entry.source_path).is_dir()
        assert all((ROOT / path).is_file() for path in entry.license_files)
        assert all((ROOT / path).is_file() for path in entry.notice_files)
        assert entry.adaptation_metadata is not None
        assert (ROOT / entry.adaptation_metadata).is_file()
    assert (ROOT / "third_party/colibri/source/THIRD_PARTY_NOTICES.md").is_file()
    assert all(not entry.notice_files for entry in manifest.entries if entry.id != "colibri")


def test_snapshot_digests_are_stable_and_manifest_digest_is_distinct() -> None:
    manifest = load_source_manifest(ROOT)
    assert manifest_identity_digest(ROOT) == manifest_identity_digest(ROOT)
    assert len(manifest_identity_digest(ROOT)) == 64
    for entry in manifest.entries:
        assert vendor_snapshot_digest(ROOT, entry.source_path) == entry.current_april_vendor_digest
        assert entry.original_upstream_snapshot_digest


def test_third_party_sources_are_not_python_production_packages() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packages = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "third_party" not in packages
    assert all(not entry.production_dependency for entry in load_source_manifest(ROOT).entries)


def test_manifest_rejects_paths_outside_third_party(tmp_path: Path) -> None:
    payload = _entry(source_path="third_party/../outside")
    path = _manifest(tmp_path, payload)
    with pytest.raises(ThirdPartySourceError, match="outside"):
        load_source_manifest(tmp_path, path)


def test_vendored_entry_does_not_require_a_fabricated_notice(tmp_path: Path) -> None:
    source = tmp_path / "third_party" / "colibri" / "source"
    (source / "c").mkdir(parents=True)
    (source / "c" / "Makefile").write_text("qwen36:\n", encoding="utf-8")
    (source / "LICENSE").write_text("license", encoding="utf-8")
    adaptation = tmp_path / "third_party" / "colibri" / "APRIL_ADAPTATION.md"
    adaptation.write_text("adaptation", encoding="utf-8")
    entry = _entry(
        status="vendored",
        upstream={"name": "Colibri", "revision": "a" * 40},
        license_files=["third_party/colibri/source/LICENSE"],
        notice_files=[],
        adaptation_metadata="third_party/colibri/APRIL_ADAPTATION.md",
    )
    digest = vendor_snapshot_digest(tmp_path, Path(entry["source_path"]))
    entry["original_upstream_snapshot_digest"] = digest
    entry["current_april_vendor_digest"] = digest
    result = inspect_source_manifest(tmp_path, _manifest(tmp_path, entry))
    assert result["ready"] is True
    assert result["entries"][0]["notice_required"] is False
    assert result["entries"][0]["license_notice_present"] is True


def test_vendored_entry_with_listed_missing_notice_is_not_ready(tmp_path: Path) -> None:
    source = tmp_path / "third_party" / "colibri" / "source"
    source.mkdir(parents=True)
    (source / "LICENSE").write_text("L", encoding="utf-8")
    (tmp_path / "third_party" / "colibri" / "APRIL_ADAPTATION.md").write_text("A", encoding="utf-8")
    entry = _entry(
        status="vendored",
        upstream={"name": "Colibri", "revision": "b" * 40},
        license_files=["third_party/colibri/source/LICENSE"],
        notice_files=["third_party/colibri/source/NOTICE"],
        adaptation_metadata="third_party/colibri/APRIL_ADAPTATION.md",
    )
    digest = vendor_snapshot_digest(tmp_path, Path(entry["source_path"]))
    entry.update(original_upstream_snapshot_digest=digest, current_april_vendor_digest=digest)
    result = inspect_source_manifest(tmp_path, _manifest(tmp_path, entry))
    assert result["ready"] is False
    assert result["entries"][0]["notice_required"] is True


def test_expected_colibri_build_artifact_is_allowed_and_outside_digest(tmp_path: Path) -> None:
    source = tmp_path / "third_party" / "colibri" / "source"
    (source / "c").mkdir(parents=True)
    (source / "LICENSE").write_text("L", encoding="utf-8")
    (tmp_path / "third_party" / "colibri" / "APRIL_ADAPTATION.md").write_text("A", encoding="utf-8")
    entry = _entry(
        status="vendored",
        upstream={"name": "Colibri", "revision": "a" * 40},
        license_files=["third_party/colibri/source/LICENSE"],
        adaptation_metadata="third_party/colibri/APRIL_ADAPTATION.md",
        allowed_local_build_artifacts=["c/qwen36"],
    )
    digest_before = vendor_snapshot_digest(
        tmp_path,
        Path(entry["source_path"]),
        allowed_local_build_artifacts=("c/qwen36",),
    )
    entry.update(
        original_upstream_snapshot_digest=digest_before,
        current_april_vendor_digest=digest_before,
    )
    manifest = _manifest(tmp_path, entry)
    absent = inspect_source_manifest(tmp_path, manifest)
    assert absent["ready"] is True
    assert absent["entries"][0]["local_build_artifacts"] == []

    binary = source / "c" / "qwen36"
    binary.write_bytes(b"\x7fELF local build output")
    present = inspect_source_manifest(tmp_path, manifest)
    assert present["ready"] is True
    assert present["entries"][0]["source_integrity"] == "ready"
    assert present["entries"][0]["local_build_artifacts"] == ["c/qwen36"]
    assert present["entries"][0]["forbidden_paths"] == []
    assert (
        vendor_snapshot_digest(
            tmp_path,
            Path(entry["source_path"]),
            allowed_local_build_artifacts=("c/qwen36",),
        )
        == digest_before
    )


def test_build_colibri_uses_upstream_make_target(monkeypatch) -> None:
    calls: list[tuple[list[str], Path, bool]] = []
    monkeypatch.setattr(
        "apps.runner.third_party_sources.shutil.which", lambda name: "/usr/bin/make"
    )

    def fake_run(argv, *, check, cwd):
        calls.append((argv, cwd, check))

    monkeypatch.setattr("apps.runner.third_party_sources.subprocess.run", fake_run)
    result = build_colibri(ROOT)
    assert result["built"] is True
    assert calls == [
        (
            [
                "make",
                "-C",
                str(ROOT / "third_party/colibri/source/c"),
                "qwen36",
                "ARCH=native",
            ],
            ROOT,
            True,
        )
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"engine": "arbitrary"}, "qwen36"), ({"arch": "native;touch"}, "native")],
)
def test_build_colibri_rejects_unallowlisted_make_inputs(kwargs, message: str) -> None:
    with pytest.raises(ThirdPartySourceError, match=message):
        build_colibri(ROOT, **kwargs)


def test_build_colibri_requires_make(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("apps.runner.third_party_sources.shutil.which", lambda name: None)
    with pytest.raises(ThirdPartySourceError, match="Make"):
        build_colibri(ROOT)


def test_colibri_python_resolution_accepts_current_compatible_interpreter() -> None:
    interpreter = resolve_colibri_python()
    assert interpreter == Path(__import__("sys").executable)


def test_colibri_python_resolution_rejects_old_and_accepts_compatible(monkeypatch) -> None:
    versions = {"old-python": "(3, 9)", "new-python": "(3, 10)"}

    def fake_run(argv, **kwargs):
        return type("Result", (), {"stdout": versions[argv[0]]})()

    monkeypatch.setattr("apps.runner.third_party_sources.subprocess.run", fake_run)
    with pytest.raises(ThirdPartySourceError, match=r"Python 3\.10"):
        resolve_colibri_python(["old-python"])
    assert resolve_colibri_python(["new-python"]) == Path("new-python")


def test_colibri_launcher_uses_fixed_argv_and_no_shell(monkeypatch) -> None:
    calls: list[tuple[list[str], Path, bool]] = []
    monkeypatch.setattr(
        "apps.runner.third_party_sources.resolve_colibri_python",
        lambda: Path("/usr/bin/python3"),
    )

    def fake_run(argv, *, cwd, check):
        calls.append((argv, cwd, check))
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("apps.runner.third_party_sources.subprocess.run", fake_run)
    assert (
        run_colibri_command(
            ROOT,
            "info",
            model=Path("/tmp/model"),
        )
        == 0
    )
    assert calls == [
        (
            [
                "/usr/bin/python3",
                str(ROOT / "third_party/colibri/source/c/coli"),
                "info",
                "--model",
                "/tmp/model",
            ],
            ROOT,
            False,
        )
    ]


def test_doctor_detects_modified_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "third_party" / "colibri" / "source"
    source.mkdir(parents=True)
    (source / "LICENSE").write_text("L", encoding="utf-8")
    (tmp_path / "third_party" / "colibri" / "APRIL_ADAPTATION.md").write_text("A", encoding="utf-8")
    entry = _entry(
        status="vendored",
        upstream={"name": "Colibri", "revision": "b" * 40},
        license_files=["third_party/colibri/source/LICENSE"],
        adaptation_metadata="third_party/colibri/APRIL_ADAPTATION.md",
    )
    digest = vendor_snapshot_digest(tmp_path, Path(entry["source_path"]))
    entry.update(original_upstream_snapshot_digest=digest, current_april_vendor_digest=digest)
    manifest = _manifest(tmp_path, entry)
    (source / "changed.py").write_text("changed", encoding="utf-8")
    result = inspect_source_manifest(tmp_path, manifest)
    assert result["ready"] is False
    assert result["entries"][0]["snapshot_matches"] is False


def test_source_hygiene_rejects_generated_and_model_artifacts() -> None:
    assert forbidden_reason("third_party/colibri/build/log.txt")
    assert forbidden_reason("third_party/colibri/source/c/qwen36")
    assert forbidden_reason("third_party/colibri/source/c/x.o")
    assert forbidden_reason("third_party/colibri/source/c/.coli_usage")
    assert forbidden_reason("third_party/colibri/models/model.gguf")
    assert forbidden_reason("third_party/reference_sources/foo/weights.bin")
    assert forbidden_reason("third_party/colibri/source/Makefile") is None
    assert forbidden_reason("third_party/colibri/APRIL_ADAPTATION.md") is None


def test_doctor_rejects_unexpected_binary_suffixes_and_runtime_cache(tmp_path: Path) -> None:
    source = tmp_path / "third_party" / "colibri" / "source"
    source.mkdir(parents=True)
    (source / "LICENSE").write_text("L", encoding="utf-8")
    (tmp_path / "third_party" / "colibri" / "APRIL_ADAPTATION.md").write_text("A", encoding="utf-8")
    entry = _entry(
        status="vendored",
        upstream={"name": "Colibri", "revision": "a" * 40},
        license_files=["third_party/colibri/source/LICENSE"],
        adaptation_metadata="third_party/colibri/APRIL_ADAPTATION.md",
        allowed_local_build_artifacts=["c/qwen36"],
    )
    digest = vendor_snapshot_digest(
        tmp_path,
        Path(entry["source_path"]),
        allowed_local_build_artifacts=("c/qwen36",),
    )
    entry.update(original_upstream_snapshot_digest=digest, current_april_vendor_digest=digest)
    manifest = _manifest(tmp_path, entry)
    (source / "unexpected").write_bytes(b"\x7fELF")
    (source / "unexpected.o").write_bytes(b"object")
    (source / "model.safetensors").write_bytes(b"weights")
    (source / ".coli_usage").write_text("cache", encoding="utf-8")
    result = inspect_source_manifest(tmp_path, manifest)
    assert result["ready"] is False
    assert result["entries"][0]["source_integrity"] == "not_ready"
    assert result["entries"][0]["snapshot_matches"] is False
    assert set(result["entries"][0]["forbidden_paths"]) >= {
        "unexpected.o",
        "model.safetensors",
        ".coli_usage",
    }


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
        json.dumps(
            {
                "manifest_id": "test",
                "schema_version": "1",
                "provenance_status": "complete",
                "entries": list(entries),
            }
        ),
        encoding="utf-8",
    )
    return target


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("[1]", "must be an object"),
        (
            json.dumps(
                {
                    "manifest_id": "x",
                    "schema_version": "1",
                    "provenance_status": "complete",
                    "entries": [1],
                }
            ),
            "invalid entry",
        ),
        (
            json.dumps(
                {
                    "manifest_id": "x",
                    "schema_version": "1",
                    "provenance_status": "complete",
                    "entries": [],
                }
            ),
            "no entries",
        ),
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


def test_manifest_rejects_missing_provenance_and_incomplete_entry(tmp_path: Path) -> None:
    missing_status = tmp_path / "third_party" / "source-manifest.json"
    missing_status.parent.mkdir(parents=True)
    missing_status.write_text(
        json.dumps({"manifest_id": "x", "schema_version": "1", "entries": []}),
        encoding="utf-8",
    )
    with pytest.raises(ThirdPartySourceError, match="provenance"):
        load_source_manifest(tmp_path)
    with pytest.raises(ThirdPartySourceError, match="incomplete"):
        load_source_manifest(
            tmp_path,
            _manifest(
                tmp_path,
                {"id": "colibri", "kind": "runtime", "status": "metadata_only"},
            ),
        )


def test_build_colibri_reports_missing_makefile(tmp_path: Path) -> None:
    source = tmp_path / "third_party" / "colibri" / "source"
    source.mkdir(parents=True)
    (source / "LICENSE").write_text("L", encoding="utf-8")
    (tmp_path / "third_party" / "colibri" / "APRIL_ADAPTATION.md").write_text("A", encoding="utf-8")
    entry = _entry(
        status="vendored",
        upstream={"name": "Colibri", "revision": "a" * 40},
        license_files=["third_party/colibri/source/LICENSE"],
        adaptation_metadata="third_party/colibri/APRIL_ADAPTATION.md",
    )
    digest = vendor_snapshot_digest(tmp_path, Path(entry["source_path"]))
    entry.update(original_upstream_snapshot_digest=digest, current_april_vendor_digest=digest)
    with pytest.raises(ThirdPartySourceError, match="Makefile"):
        build_colibri(tmp_path, path=_manifest(tmp_path, entry))


def test_doctor_reports_nested_git_and_generated_paths(tmp_path: Path) -> None:
    source = tmp_path / "third_party" / "colibri" / "source"
    source.mkdir(parents=True)
    (source / "LICENSE").write_text("L", encoding="utf-8")
    (source / ".git").mkdir()
    (source / "build").mkdir()
    (source / "build" / "output").write_text("x", encoding="utf-8")
    entry = _entry(
        status="vendored",
        upstream={"name": "Colibri", "revision": "a" * 40},
        license_files=["third_party/colibri/source/LICENSE"],
        adaptation_metadata="third_party/colibri/APRIL_ADAPTATION.md",
    )
    digest = vendor_snapshot_digest(tmp_path, Path(entry["source_path"]))
    entry.update(original_upstream_snapshot_digest=digest, current_april_vendor_digest=digest)
    result = inspect_source_manifest(tmp_path, _manifest(tmp_path, entry))
    assert result["ready"] is False
    assert result["entries"][0]["nested_git"] is True
    assert result["entries"][0]["forbidden_paths"]


def test_colibri_launcher_and_build_errors_are_wrapped(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ThirdPartySourceError, match="Unsupported"):
        run_colibri_command(tmp_path, "unknown", model=Path("model"))

    monkeypatch.setattr(
        "apps.runner.third_party_sources.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("blocked")),
    )
    with pytest.raises(ThirdPartySourceError, match="could not start"):
        _run_local_build(["make"], cwd=tmp_path)
    monkeypatch.setattr(
        "apps.runner.third_party_sources.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(7, ["make"])),
    )
    with pytest.raises(ThirdPartySourceError, match="exit code 7"):
        _run_local_build(["make"], cwd=tmp_path)


def test_path_helpers_fail_closed_and_tree_details_handle_missing_paths(tmp_path: Path) -> None:
    with pytest.raises(ThirdPartySourceError, match="escapes"):
        _safe_relative_path(tmp_path, Path("../outside"))
    assert _safe_is_dir(tmp_path, Path("../outside")) is False
    assert _safe_is_file(tmp_path, Path("../outside")) is False
    assert _vendor_tree_details(tmp_path, Path("../outside"))[2]
    missing_file = tmp_path / "file"
    missing_file.write_text("x", encoding="utf-8")
    assert _vendor_tree_details(tmp_path, Path("file"))[0] is None


@pytest.mark.parametrize(
    ("entry_updates", "reason"),
    [
        ({"status": "metadata_only"}, "source_not_staged"),
        ({"status": "vendored"}, "source_directory_missing"),
    ],
)
def test_doctor_reports_provenance_and_missing_source_reasons(
    tmp_path: Path, entry_updates: dict[str, object], reason: str
) -> None:
    entry = _entry(**entry_updates)
    if entry_updates["status"] == "vendored":
        entry.update(
            original_upstream_snapshot_digest="a" * 64,
            current_april_vendor_digest="a" * 64,
        )
    result = inspect_source_manifest(tmp_path, _manifest(tmp_path, entry))
    assert result["entries"][0]["reason"] == reason


def test_doctor_reports_missing_license_adaptation_and_forbidden_artifact(tmp_path: Path) -> None:
    source = tmp_path / "third_party" / "colibri" / "source"
    source.mkdir(parents=True)
    (source / "artifact.o").write_bytes(b"object")
    entry = _entry(
        status="vendored",
        upstream={"name": "Colibri", "revision": "a" * 40},
        adaptation_metadata="third_party/colibri/APRIL_ADAPTATION.md",
    )
    digest = vendor_snapshot_digest(tmp_path, Path(entry["source_path"]))
    entry.update(original_upstream_snapshot_digest=digest, current_april_vendor_digest=digest)
    result = inspect_source_manifest(tmp_path, _manifest(tmp_path, entry))
    assert result["entries"][0]["reason"] == "license_missing"
    (source / "LICENSE").write_text("L", encoding="utf-8")
    entry["license_files"] = ["third_party/colibri/source/LICENSE"]
    digest = vendor_snapshot_digest(tmp_path, Path(entry["source_path"]))
    entry.update(original_upstream_snapshot_digest=digest, current_april_vendor_digest=digest)
    result = inspect_source_manifest(tmp_path, _manifest(tmp_path, entry))
    assert result["entries"][0]["reason"] == "adaptation_metadata_missing"
    (tmp_path / "third_party" / "colibri" / "APRIL_ADAPTATION.md").write_text("A", encoding="utf-8")
    result = inspect_source_manifest(tmp_path, _manifest(tmp_path, entry))
    assert result["entries"][0]["reason"] == "generated_or_model_artifact_present"


def test_launcher_reports_missing_script_and_start_failure(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "third_party" / "colibri" / "source"
    source.mkdir(parents=True)
    (source / "LICENSE").write_text("L", encoding="utf-8")
    (tmp_path / "third_party" / "colibri" / "APRIL_ADAPTATION.md").write_text("A", encoding="utf-8")
    entry = _entry(
        status="vendored",
        upstream={"name": "Colibri", "revision": "a" * 40},
        license_files=["third_party/colibri/source/LICENSE"],
        adaptation_metadata="third_party/colibri/APRIL_ADAPTATION.md",
        allowed_local_build_artifacts=["c/qwen36"],
    )
    digest = vendor_snapshot_digest(tmp_path, Path(entry["source_path"]))
    entry.update(original_upstream_snapshot_digest=digest, current_april_vendor_digest=digest)
    manifest = _manifest(tmp_path, entry)
    with pytest.raises(ThirdPartySourceError, match="launcher c/coli"):
        run_colibri_command(tmp_path, "info", model=Path("model"))

    script = source / "c" / "coli"
    script.parent.mkdir()
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (script.parent / "qwen36").write_bytes(b"\x7fELF local build output")
    monkeypatch.setattr(
        "apps.runner.third_party_sources.resolve_colibri_python",
        lambda: Path("python3"),
    )
    monkeypatch.setattr(
        "apps.runner.third_party_sources.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("blocked")),
    )
    digest = vendor_snapshot_digest(
        tmp_path,
        Path(entry["source_path"]),
        allowed_local_build_artifacts=("c/qwen36",),
    )
    entry.update(original_upstream_snapshot_digest=digest, current_april_vendor_digest=digest)
    _manifest(tmp_path, entry)
    with pytest.raises(ThirdPartySourceError, match="could not start"):
        run_colibri_command(tmp_path, "info", model=Path("model"))
    assert manifest.exists()


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
        (_entry(allowed_local_build_artifacts=["../qwen36"]), "local build artifact"),
        (
            _entry(kind="reference", allowed_local_build_artifacts=["bin/tool"]),
            "local build artifacts",
        ),
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


def test_manifest_colibri_property_fails_without_colibri() -> None:
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
        original_upstream_snapshot_digest=None,
        current_april_vendor_digest=None,
        snapshot_exclusions=(),
    )
    with pytest.raises(ThirdPartySourceError, match="Colibri"):
        _ = ThirdPartyManifest("1", "test", "complete", (entry,)).colibri


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
