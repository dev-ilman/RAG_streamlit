from __future__ import annotations

import pytest

from app.services.extraction import (
    UnsupportedDocumentError,
    extract_text,
    guess_mime_type,
    html_to_text,
    normalise_text,
)


def test_extension_beats_a_generic_declared_type() -> None:
    assert guess_mime_type("notes.md", "application/octet-stream") == "text/markdown"


def test_declared_type_used_when_filename_is_unhelpful() -> None:
    assert guess_mime_type(None, "text/plain; charset=utf-8") == "text/plain"


def test_unknown_type_is_rejected() -> None:
    with pytest.raises(UnsupportedDocumentError):
        guess_mime_type("archive.zip", "application/zip")


def test_html_scripts_and_styles_are_dropped() -> None:
    html = (
        "<html><head><style>body{color:red}</style></head>"
        "<body><h1>Title</h1><script>alert(1)</script>"
        "<p>Real content here.</p></body></html>"
    )
    text = html_to_text(html)
    assert "Real content here." in text
    assert "alert" not in text
    assert "color:red" not in text


def test_entities_are_decoded() -> None:
    assert "A & B" in html_to_text("<p>A &amp; B</p>")


def test_normalise_collapses_whitespace_and_blank_runs() -> None:
    assert normalise_text("a\r\n\n\n\nb   c\xa0d") == "a\n\nb c d"


def test_extract_text_handles_plain_utf8() -> None:
    assert extract_text("héllo wörld".encode(), "text/plain") == "héllo wörld"


def test_extract_text_rejects_unsupported_mime() -> None:
    with pytest.raises(UnsupportedDocumentError):
        extract_text(b"data", "application/zip")
