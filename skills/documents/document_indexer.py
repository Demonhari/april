from __future__ import annotations

import hashlib
from typing import Any

from april_common.path_security import MODEL_SUFFIXES, normalize_existing_path
from april_common.settings import get_settings
from april_common.time import utc_now_iso
from services.memory.vector_memory import VectorMemory
from skills.base import timed_tool
from skills.code.repo_indexer import chunk_text
from skills.documents.extractors import ExtractedDocument, UnsupportedDocument, extract_document
from skills.filesystem.common import current_path_policy, ignored, read_gitignore_patterns
from skills.schemas import ToolDefinition, ToolResult


async def document_indexer(args: dict[str, Any]) -> ToolResult:
    async def run() -> ToolResult:
        settings = get_settings()
        policy = current_path_policy()
        root = normalize_existing_path(args["folder_path"], policy)
        patterns = read_gitignore_patterns(root)
        vector = VectorMemory(settings.vector_index_path)
        indexed = 0
        chunks: list[tuple[str, str, int | None, int | None]] = []
        documents: list[dict[str, object]] = []
        unsupported: list[dict[str, str]] = []
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            if ignored(file_path, root=root, patterns=patterns):
                if file_path.suffix.lower() in MODEL_SUFFIXES:
                    unsupported.append(
                        {
                            "path": str(file_path),
                            "reason": "Unsupported binary/model artifact format.",
                        }
                    )
                continue
            extracted = extract_document(file_path, max_bytes=min(policy.max_read_bytes, 500_000))
            if isinstance(extracted, UnsupportedDocument):
                unsupported.append({"path": extracted.source_path, "reason": extracted.reason})
                continue
            assert isinstance(extracted, ExtractedDocument)
            chunk_count = 0
            lines = extracted.content.splitlines()
            for content, start, end in chunk_text(lines):
                if content.strip():
                    chunks.append((str(file_path), content, start, end))
                    indexed += 1
                    chunk_count += 1
            documents.append(
                {
                    "identifier": extracted.identifier,
                    "source_path": extracted.source_path,
                    "content_hash": extracted.content_hash,
                    "extraction_type": extracted.extraction_type,
                    "chunk_count": chunk_count,
                    "indexed_at": utc_now_iso(),
                }
            )
        source_id = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
        vector.index_chunks(
            source_type="document",
            source_id=source_id,
            project_id=None,
            chunks=chunks,
        )
        return ToolResult(
            ok=True,
            stdout=f"Indexed {indexed} document chunks.",
            data={
                "chunks": indexed,
                "path": str(root),
                "documents": documents,
                "unsupported": unsupported,
            },
            risk_level="safe_write",
            permission_level=2,
        )

    return await timed_tool(run, risk_level="safe_write", permission_level=2)


def document_indexer_definition() -> ToolDefinition:
    return ToolDefinition(
        name="document_indexer",
        description="Index local text documents and optionally PDFs into the local vector store.",
        permission_level=2,
        risk_level="safe_write",
        allowed_agents={"reading_agent"},
        executor=document_indexer,
        affected_paths=lambda args: [str(args.get("folder_path", ""))],
    )
