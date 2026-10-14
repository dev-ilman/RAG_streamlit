"""Pure-function tests for the splitter. No database, no network."""

from __future__ import annotations

import pytest

from app.services.chunking import split_text


def test_empty_text_produces_no_chunks() -> None:
    assert split_text("   \n\n  ") == []


def test_short_text_is_a_single_chunk() -> None:
    chunks = split_text("A single short sentence.", chunk_size=1000, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == "A single short sentence."
    assert chunks[0].index == 0


def test_chunks_respect_the_size_limit() -> None:
    text = "Sentence number {} about battery degradation. ".format
    body = "".join(text(i) for i in range(200))
    chunks = split_text(body, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    # Packing may exceed the target slightly when a single piece is oversized,
    # but never by more than one piece plus the overlap carried forward.
    assert all(len(c.text) <= 300 + 50 for c in chunks)


def test_indices_are_contiguous_and_zero_based() -> None:
    body = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\n" * 40
    chunks = split_text(body, chunk_size=200, chunk_overlap=40)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_carries_context_across_the_boundary() -> None:
    body = " ".join(f"word{i}" for i in range(400))
    chunks = split_text(body, chunk_size=200, chunk_overlap=60)
    assert len(chunks) > 2
    # The tail of one chunk should reappear at the head of the next.
    tail_word = chunks[0].text.split()[-1]
    assert tail_word in chunks[1].text


def test_text_with_no_separator_is_hard_split() -> None:
    chunks = split_text("x" * 700, chunk_size=300, chunk_overlap=0)
    assert [len(c.text) for c in chunks] == [300, 300, 100]


def test_overlap_larger_than_chunk_is_clamped() -> None:
    # Would otherwise loop forever: the cursor could never advance.
    chunks = split_text("word " * 500, chunk_size=100, chunk_overlap=500)
    assert len(chunks) > 1


@pytest.mark.parametrize("bad_size", [0, -1])
def test_invalid_chunk_size_is_rejected(bad_size: int) -> None:
    with pytest.raises(ValueError):
        split_text("some text", chunk_size=bad_size)


def test_token_estimate_is_positive() -> None:
    chunks = split_text("hello world " * 50, chunk_size=200, chunk_overlap=20)
    assert all(c.token_estimate >= 1 for c in chunks)
