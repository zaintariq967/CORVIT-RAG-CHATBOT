"""
pdf_processor.py
------------------------------------------------------------------
Stage 1 of the RAG pipeline: turn the Corvit PDF into clean, small,
retrievable chunks of text.

Design choice: SECTION-AWARE CHUNKING, not blind fixed-size splitting.
The source PDF is written as numbered sections ("1. Overview",
"2. Vision and Training Delivery", ...), so we split on those headers
first -- each chunk then stays on a single topic (fees, campuses,
courses, etc.), which makes retrieval far more precise than chopping
the document into arbitrary 500-character windows.

Any section that is still too long for a single embedding chunk falls
back to a sliding window with overlap, so no chunk grows unbounded.
"""

import re
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader

# Matches numbered section headers such as "12. Rawalpindi Campus" that
# start their own line -- exactly how Corvit's knowledge-base PDF is
# structured. Verified against the actual extracted text of corvit.pdf.
SECTION_PATTERN = re.compile(r"(?m)^(\d{1,2})\.\s+([A-Z][^\n]{2,80})\n")

DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP = 150


@dataclass
class Chunk:
    """One retrievable unit of text plus light traceability metadata."""

    chunk_id: int
    text: str
    section_title: str
    page_number: int


def extract_pages(pdf_path: str) -> List[str]:
    """Return the raw text of each page, in order."""
    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]


def _sliding_window(text: str, max_chars: int, overlap: int) -> List[str]:
    """Fallback splitter for any section longer than max_chars."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    pieces = []
    start = 0
    while start < len(text):
        end = start + max_chars
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return pieces


def chunk_document(
    pdf_path: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Chunk]:
    """
    Extract + chunk a PDF into a list of Chunk objects.

    Strategy:
      1. Concatenate all page text, remembering where each page starts
         so every chunk can still be traced back to a page number.
      2. Split on numbered section headers.
      3. Sliding-window any individual section that is still too long.
      4. If no numbered headers are found at all (e.g. a different PDF
         was dropped into assets/data/), fall back to sliding-window
         chunking over the whole document so the app still works.
    """
    pages = extract_pages(pdf_path)

    full_text = ""
    page_starts = []  # (char_offset, page_number)
    for page_num, page_text in enumerate(pages, start=1):
        page_starts.append((len(full_text), page_num))
        full_text += page_text + "\n"

    def page_for_offset(offset: int) -> int:
        page = 1
        for start_offset, num in page_starts:
            if offset >= start_offset:
                page = num
            else:
                break
        return page

    matches = list(SECTION_PATTERN.finditer(full_text))
    chunks: List[Chunk] = []
    next_id = 0

    if not matches:
        for piece in _sliding_window(full_text, max_chars, overlap):
            chunks.append(Chunk(next_id, piece, "Document", 1))
            next_id += 1
        return chunks

    for i, m in enumerate(matches):
        section_start = m.start()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        section_title = f"{m.group(1)}. {m.group(2)}"
        section_text = full_text[section_start:section_end]
        page_num = page_for_offset(section_start)

        for piece in _sliding_window(section_text, max_chars, overlap):
            chunks.append(Chunk(next_id, piece, section_title, page_num))
            next_id += 1

    return chunks