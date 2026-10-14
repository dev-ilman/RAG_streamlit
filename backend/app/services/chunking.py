"""Recursive character splitting.

A direct, dependency-free port of the behaviour the notebook relied on from
LangChain's ``RecursiveCharacterTextSplitter``: try to break on the largest
natural boundary that fits, and only fall back to hard character cuts when a
single run of text has no boundary at all.

The notebook used ``chunk_size=200`` with ``chunk_overlap=50``. That is fine
for a demo over one page of text, but it fragments sentences and multiplies
embedding calls; the defaults here (1000 / 150) keep passages coherent enough
to cite while still fitting comfortably in an embedding window.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", " ", "")


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    text: str
    start_char: int

    @property
    def token_estimate(self) -> int:
        """Rough token count. Good enough for budgeting, not for billing."""
        return max(1, len(self.text) // 4)


def _split_on(text: str, separators: tuple[str, ...], chunk_size: int) -> list[str]:
    """Break ``text`` into pieces no larger than ``chunk_size`` where possible."""
    if len(text) <= chunk_size:
        return [text] if text else []

    separator, *rest = separators
    remaining = tuple(rest)

    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(separator)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        piece = part + (separator if i < len(parts) - 1 else "")
        if not piece:
            continue
        if len(piece) > chunk_size and remaining:
            pieces.extend(_split_on(piece, remaining, chunk_size))
        else:
            pieces.append(piece)
    return pieces


def split_text(
    text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks, preserving source offsets."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        # An overlap at or above the window would never advance the cursor.
        chunk_overlap = chunk_size // 4

    normalised = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalised:
        return []

    pieces = _split_on(normalised, separators, chunk_size)

    # Greedily pack pieces up to chunk_size, then carry the tail forward as the
    # overlap so a fact spanning a boundary still appears whole in one chunk.
    chunks: list[Chunk] = []
    buffer = ""
    cursor = 0

    def flush(buf: str, start: int) -> None:
        cleaned = buf.strip()
        if cleaned:
            chunks.append(Chunk(index=len(chunks), text=cleaned, start_char=start))

    for piece in pieces:
        if buffer and len(buffer) + len(piece) > chunk_size:
            flush(buffer, cursor)
            tail = buffer[-chunk_overlap:] if chunk_overlap else ""
            cursor += max(1, len(buffer) - len(tail))
            buffer = tail + piece
        else:
            buffer += piece

    flush(buffer, cursor)
    return chunks
