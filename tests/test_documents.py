from __future__ import annotations

import sys
import types

import anyio
import pytest
from fastapi.testclient import TestClient

from april_common.errors import PermissionDeniedError
from services.api.server import create_app
from services.memory.vector_memory import VectorMemory
from skills.documents.document_indexer import document_indexer
from skills.documents.document_search import document_search
from tests.test_core_api import auth, make_container


def _make_corpus(settings_tmp) -> object:
    folder = settings_tmp.home / "docs"
    folder.mkdir()
    (folder / "guide.md").write_text("# Local guide\nanimation pipeline notes\n", encoding="utf-8")
    (folder / "notes.txt").write_text("plain text reading notes\n", encoding="utf-8")
    return folder


def test_document_indexer_indexes_text_files(settings_tmp) -> None:
    folder = _make_corpus(settings_tmp)
    result = anyio.run(document_indexer, {"folder_path": str(folder)})
    assert result.ok
    assert result.data["chunks"] > 0
    assert result.permission_level == 2
    assert result.risk_level == "safe_write"


def test_document_indexer_skips_binary_and_rejects_out_of_root(settings_tmp) -> None:
    folder = _make_corpus(settings_tmp)
    (folder / "blob.bin").write_bytes(b"\x00\x01binary\x00payload")

    result = anyio.run(document_indexer, {"folder_path": str(folder)})
    assert result.ok
    assert result.data["unsupported"]
    assert any(item["path"].endswith("blob.bin") for item in result.data["unsupported"])
    vector = VectorMemory(settings_tmp.vector_index_path)
    indexed_paths = {
        path for source in vector.sources(source_type="document") for path in source["paths"]
    }
    assert not any(path.endswith("blob.bin") for path in indexed_paths)
    assert any(path.endswith("guide.md") for path in indexed_paths)

    with pytest.raises(PermissionDeniedError):
        anyio.run(document_indexer, {"folder_path": str(settings_tmp.home.parent)})


def test_document_indexer_extracts_pdf_when_optional_dependency_exists(
    settings_tmp, monkeypatch
) -> None:
    folder = settings_tmp.home / "pdfs"
    folder.mkdir()
    pdf = folder / "guide.pdf"
    pdf.write_bytes(b"%PDF local fixture")

    class FakePage:
        def extract_text(self) -> str:
            return "pdf animation notes"

    class FakeReader:
        def __init__(self, path: str) -> None:
            assert path == str(pdf)
            self.pages = [FakePage()]

    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=FakeReader))
    result = anyio.run(document_indexer, {"folder_path": str(folder)})
    assert result.ok
    assert result.data["chunks"] == 1
    assert result.data["documents"][0]["extraction_type"] == "pdf"
    assert result.data["unsupported"] == []

    vector = VectorMemory(settings_tmp.vector_index_path)
    chunks = vector.search("animation", source_type="document")
    assert chunks
    assert chunks[0].metadata["path"] == str(pdf)


def test_document_reindex_removes_deleted_files(settings_tmp) -> None:
    folder = settings_tmp.home / "docs-delete"
    folder.mkdir()
    note = folder / "note.txt"
    note.write_text("temporary document text\n", encoding="utf-8")
    first = anyio.run(document_indexer, {"folder_path": str(folder)})
    assert first.data["chunks"] == 1
    note.unlink()
    second = anyio.run(document_indexer, {"folder_path": str(folder)})
    assert second.data["chunks"] == 0
    vector = VectorMemory(settings_tmp.vector_index_path)
    assert vector.sources(source_type="document") == []


def test_document_search_and_retriever_carry_line_metadata(settings_tmp) -> None:
    vector = VectorMemory(settings_tmp.vector_index_path)
    doc_path = str(settings_tmp.home / "docs" / "guide.md")
    vector.index_chunks(
        source_type="document",
        source_id="docs",
        project_id=None,
        chunks=[(doc_path, "animation pipeline notes", 1, 2)],
    )

    result = anyio.run(document_search, {"query": "animation", "limit": 5})
    assert result.ok
    assert result.permission_level == 1
    assert result.risk_level == "read_only"
    top = result.data["chunks"][0]
    assert top["path"] == doc_path
    assert top["start_line"] == 1
    assert top["end_line"] == 2

    container = anyio.run(make_container, settings_tmp)
    container.vector_memory.index_chunks(
        source_type="document",
        source_id="docs",
        project_id=None,
        chunks=[(doc_path, "animation pipeline notes", 1, 2)],
    )
    chunks = container.memory_retriever.document_chunks("animation")
    assert chunks
    assert chunks[0].metadata["path"] == doc_path
    assert chunks[0].metadata["start_line"] == 1
    assert chunks[0].metadata["end_line"] == 2


def test_reading_chat_includes_document_context_and_citation(settings_tmp, monkeypatch) -> None:
    monkeypatch.setenv("APRIL_LEGACY_ORCHESTRATOR", "1")
    container = anyio.run(make_container, settings_tmp)
    doc_path = str(settings_tmp.home / "docs" / "guide.md")
    container.vector_memory.index_chunks(
        source_type="document",
        source_id="docs",
        project_id=None,
        chunks=[(doc_path, "animation pipeline notes", 1, 2)],
    )
    client = TestClient(create_app(container))
    response = client.post(
        "/chat",
        json={"message": "Summarize my indexed notes."},
        headers=auth(settings_tmp),
    )
    assert response.status_code == 200
    result = response.json()["result"]
    citation_paths = [citation["path"] for citation in result["local_citations"]]
    assert doc_path in citation_paths

    prompt = "\n".join(
        message.content
        for message in container.runtime_client.last_messages  # type: ignore[attr-defined]
    )
    assert "Indexed document chunks" in prompt
