from __future__ import annotations

import unicodedata
from collections.abc import Iterable

LEXICAL_TOKENIZER_VERSION = "unicode-nfkc-casefold-v1"
HASHED_TOKEN_IMPLEMENTATION_VERSION = "hashed-token-unicode-v2"
DEFAULT_MAX_WORD_TOKENS = 512
DEFAULT_MAX_NGRAM_TOKENS = 512
DEFAULT_NGRAM_SIZE = 3


def normalize_text(text: str) -> str:
    """Return APRIL's deterministic Unicode-normalized text representation."""
    return unicodedata.normalize("NFKC", text).casefold()


def word_tokens(text: str, *, max_tokens: int = DEFAULT_MAX_WORD_TOKENS) -> list[str]:
    """Tokenize letters, numbers, marks, and meaningful identifier underscores.

    Combining marks remain attached to their base token. Punctuation, symbols,
    whitespace, and control characters are boundaries. Leading/trailing and
    repeated underscores are discarded while embedded underscores are retained.
    """
    if max_tokens <= 0:
        return []
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if len(tokens) >= max_tokens or not current:
            current.clear()
            return
        token = "".join(current).strip("_")
        current.clear()
        while "__" in token:
            token = token.replace("__", "_")
        if token and any(_is_letter_number_or_mark(char) for char in token):
            tokens.append(token)

    for char in normalize_text(text):
        if _is_letter_number_or_mark(char) or (char == "_" and current and current[-1] != "_"):
            current.append(char)
        else:
            flush()
            if len(tokens) >= max_tokens:
                break
    flush()
    return tokens[:max_tokens]


def embedding_tokens(
    text: str,
    *,
    max_word_tokens: int = DEFAULT_MAX_WORD_TOKENS,
    max_ngram_tokens: int = DEFAULT_MAX_NGRAM_TOKENS,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
) -> list[str]:
    """Return bounded, ordered, deduplicated word and Unicode n-gram tokens."""
    words = word_tokens(text, max_tokens=max_word_tokens)
    ordered: list[str] = []
    seen: set[str] = set()
    _extend_unique(ordered, seen, words)
    if max_ngram_tokens <= 0 or ngram_size < 2:
        return ordered

    added = 0
    for token in words:
        if token.isascii():
            continue
        clusters = _grapheme_like_clusters(token)
        if len(clusters) < ngram_size:
            ngrams = [f"ng:{token}"]
        else:
            ngrams = [
                f"ng:{''.join(clusters[index : index + ngram_size])}"
                for index in range(len(clusters) - ngram_size + 1)
            ]
        for ngram in ngrams:
            if ngram in seen:
                continue
            seen.add(ngram)
            ordered.append(ngram)
            added += 1
            if added >= max_ngram_tokens:
                return ordered
    return ordered


def _grapheme_like_clusters(token: str) -> list[str]:
    clusters: list[str] = []
    for char in token:
        if unicodedata.category(char).startswith("M") and clusters:
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


def _is_letter_number_or_mark(char: str) -> bool:
    return unicodedata.category(char)[:1] in {"L", "N", "M"}


def _extend_unique(target: list[str], seen: set[str], values: Iterable[str]) -> None:
    for value in values:
        if value not in seen:
            seen.add(value)
            target.append(value)
