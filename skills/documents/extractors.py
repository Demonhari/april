from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from april_common.path_security import ensure_text_file


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    identifier: str
    source_path: str
    content: str
    content_hash: str
    extraction_type: str


@dataclass(frozen=True, slots=True)
class UnsupportedDocument:
    source_path: str
    reason: str


class DocumentExtractor(Protocol):
    extraction_type: str

    def extract(self, path: Path, *, max_bytes: int) -> ExtractedDocument | UnsupportedDocument: ...


class TextExtractor:
    extraction_type = "text"

    def extract(self, path: Path, *, max_bytes: int) -> ExtractedDocument | UnsupportedDocument:
        try:
            ensure_text_file(path, max_bytes=max_bytes)
        except Exception as exc:
            return UnsupportedDocument(source_path=str(path), reason=str(exc))
        content = path.read_text(encoding="utf-8", errors="replace")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return ExtractedDocument(
            identifier=digest,
            source_path=str(path),
            content=content,
            content_hash=digest,
            extraction_type=self.extraction_type,
        )


class PdfExtractor:
    extraction_type = "pdf"

    def extract(self, path: Path, *, max_bytes: int) -> ExtractedDocument | UnsupportedDocument:
        if path.stat().st_size > max_bytes:
            return UnsupportedDocument(source_path=str(path), reason="PDF exceeds maximum size.")
        try:
            pypdf = importlib.import_module("pypdf")
        except ImportError:
            return UnsupportedDocument(
                source_path=str(path),
                reason="PDF extraction requires optional dependency pypdf.",
            )
        try:
            reader = pypdf.PdfReader(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            return UnsupportedDocument(
                source_path=str(path), reason=f"PDF extraction failed: {exc}"
            )
        content = text.strip()
        if not content:
            return UnsupportedDocument(
                source_path=str(path), reason="PDF contained no extractable text."
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return ExtractedDocument(
            identifier=digest,
            source_path=str(path),
            content=content,
            content_hash=digest,
            extraction_type=self.extraction_type,
        )


def extract_document(path: Path, *, max_bytes: int) -> ExtractedDocument | UnsupportedDocument:
    extractor: DocumentExtractor = (
        PdfExtractor() if path.suffix.lower() == ".pdf" else TextExtractor()
    )
    return extractor.extract(path, max_bytes=max_bytes)
