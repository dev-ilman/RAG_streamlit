"""Turn uploaded bytes or a URL into plain text.

Replaces the notebook's ``PyPDFLoader`` / ``TextLoader`` / ``WebBaseLoader``.
HTML is stripped with the standard library rather than BeautifulSoup, which
removes a dependency and keeps the container image small.
"""

from __future__ import annotations

import io
import re
from html.parser import HTMLParser

import httpx
from pypdf import PdfReader

SUPPORTED_MIME_TYPES: dict[str, str] = {
    "application/pdf": "pdf",
    "text/plain": "text",
    "text/markdown": "text",
    "text/html": "html",
    "application/json": "text",
    "text/csv": "text",
}

_EXTENSION_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".json": "application/json",
    ".csv": "text/csv",
}

_WS_RE = re.compile(r"[ \t\x0b\f]+")
_BLANKS_RE = re.compile(r"\n{3,}")


class UnsupportedDocumentError(ValueError):
    """The file type cannot be parsed into text."""


class ExtractionError(RuntimeError):
    """The file type is supported but this particular file could not be read."""


def guess_mime_type(filename: str | None, declared: str | None) -> str:
    """Prefer the extension; browsers routinely send octet-stream for uploads."""
    if filename:
        lowered = filename.lower()
        for ext, mime in _EXTENSION_MIME.items():
            if lowered.endswith(ext):
                return mime
    if declared:
        base = declared.split(";", 1)[0].strip().lower()
        if base in SUPPORTED_MIME_TYPES:
            return base
    raise UnsupportedDocumentError(
        f"unsupported file type: {declared or filename or 'unknown'}. "
        f"Supported: {', '.join(sorted(SUPPORTED_MIME_TYPES))}"
    )


def normalise_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = _WS_RE.sub(" ", text)
    return _BLANKS_RE.sub("\n\n", text).strip()


class _TextExtractingParser(HTMLParser):
    """Collect visible text, dropping script/style and other non-content nodes."""

    _SKIP = frozenset({"script", "style", "noscript", "template", "svg", "head"})
    _BLOCK = frozenset(
        {"p", "div", "br", "li", "tr", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self._parts.append(data)

    @property
    def text(self) -> str:
        return normalise_text("".join(self._parts))


def html_to_text(html: str) -> str:
    parser = _TextExtractingParser()
    parser.feed(html)
    parser.close()
    return parser.text


def pdf_to_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises a wide range of parse errors
        raise ExtractionError(f"could not read PDF: {exc}") from exc
    if not any(p.strip() for p in pages):
        raise ExtractionError(
            "no selectable text found in this PDF (it is probably a scan; run OCR before uploading)"
        )
    return normalise_text("\n\n".join(pages))


def extract_text(data: bytes, mime_type: str) -> str:
    kind = SUPPORTED_MIME_TYPES.get(mime_type)
    if kind is None:
        raise UnsupportedDocumentError(f"unsupported mime type: {mime_type}")
    if kind == "pdf":
        return pdf_to_text(data)
    decoded = data.decode("utf-8", errors="replace")
    return html_to_text(decoded) if kind == "html" else normalise_text(decoded)


async def fetch_url(
    url: str, *, request_timeout: float = 30.0, max_bytes: int = 10_485_760
) -> tuple[str, str]:
    """Download a page and return ``(text, resolved_content_type)``."""
    if not url.startswith(("http://", "https://")):
        raise UnsupportedDocumentError("only http(s) URLs can be ingested")
    async with httpx.AsyncClient(
        timeout=request_timeout, follow_redirects=True, max_redirects=5
    ) as client:
        try:
            response = await client.get(url, headers={"user-agent": "rag-platform/1.0"})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExtractionError(f"could not fetch {url}: {exc}") from exc
    body = response.content[:max_bytes]
    content_type = response.headers.get("content-type", "text/html").split(";")[0].strip()
    if content_type not in SUPPORTED_MIME_TYPES:
        content_type = "text/html"
    return extract_text(body, content_type), content_type
