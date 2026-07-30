"""AST-aware Python, Markdown, and plain-text file chunking."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

Span = tuple[int, int]

DEF_NODES: tuple[type[ast.AST], ...] = (
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)
HEADING_RE = re.compile(r"^#{1,6}\s")
FENCE_RE = re.compile(r"^(```|~~~)")


class Chunk(BaseModel):
    """A single retrievable text snippet from a file.

    Invariant: ``text == source[first_character_index:last_character_index]``
    for the source the chunk was built from.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str


def line_offsets(source: str) -> list[int]:
    """Return the character index where each line starts.

    ``line_offsets(source)[i]`` is the index of the first character of
    line ``i`` (0-based). A trailing sentinel points just past the end
    of the source.
    """
    offsets = [0]
    for line in source.split("\n"):
        offsets.append(offsets[-1] + len(line) + 1)
    return offsets


def node_end_index(node: ast.stmt, offsets: list[int], source: str) -> int:
    """Return the character index just past the end of an AST node.

    ``ast`` reports ``end_col_offset`` as a *byte* offset into the
    UTF-8 encoding of the line, while ``offsets`` (and Python string
    indexing) work in *characters*. On any line containing multi-byte
    UTF-8 characters before the node's end column, using the byte
    offset directly would overshoot the true character index. We
    re-encode the line and decode only the relevant byte prefix to
    convert byte offset -> character offset correctly.
    """
    if node.end_lineno is None or node.end_col_offset is None:
        raise ValueError("Node lacks line/col location information.")

    line_start = offsets[node.end_lineno - 1]
    line_end = offsets[node.end_lineno]
    line_text = source[line_start:line_end]

    byte_prefix = line_text.encode("utf-8")[: node.end_col_offset]
    char_col = len(byte_prefix.decode("utf-8", errors="ignore"))

    return line_start + char_col


def node_span(node: ast.stmt, offsets: list[int], source: str) -> Span:
    """Return the (start, end) character span of a node, decorators included."""
    decorators = getattr(node, "decorator_list", [])
    start_line = decorators[0].lineno if decorators else node.lineno
    start = offsets[start_line - 1]
    end = node_end_index(node, offsets, source)
    return start, end


class Chunker:
    """Splits source code, Markdown, and plain-text files into `Chunk`s."""

    DEF_NODES: ClassVar[tuple[type[ast.AST], ...]] = DEF_NODES
    HEADING_RE: ClassVar[re.Pattern[str]] = HEADING_RE
    FENCE_RE: ClassVar[re.Pattern[str]] = FENCE_RE

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Args:
        max_chunk_size: Maximum allowed character length for any chunk.
        """
        self.max_chunk_size = max_chunk_size

    # ------------------------------------------------------------------
    # Generic span utilities
    # ------------------------------------------------------------------

    def split_span(self, source: str, start: int, end: int) -> list[Span]:
        """Break a window larger than `max_chunk_size` into smaller spans.

        Prefers to cut on blank lines, falling back to single newlines,
        falling back to a hard cut at `max_chunk_size` characters.
        """
        spans = []
        pos = start
        while end - pos > self.max_chunk_size:
            window = source[pos : pos + self.max_chunk_size]
            cut = window.rfind("\n\n")
            if cut != -1:
                cut += 2
            else:
                cut = window.rfind("\n") + 1
            if cut <= 0:
                cut = self.max_chunk_size

            spans.append((pos, pos + cut))
            pos += cut

        if pos < end:
            spans.append((pos, end))
        return spans

    def fill_gaps(self, spans: list[Span], source: str) -> list[Span]:
        """Ensure spans cover the whole source, merging or filling any gaps."""
        filled: list[Span] = []
        pos = 0

        for start, end in spans:
            if start > pos:
                if filled and (start - filled[-1][0]) <= self.max_chunk_size:
                    filled[-1] = (filled[-1][0], start)
                elif source[pos:start].strip():
                    filled.extend(self.split_span(source, pos, start))
            filled.append((start, end))
            pos = end

        if pos < len(source):
            if filled and (len(source) - filled[-1][0]) <= self.max_chunk_size:
                filled[-1] = (filled[-1][0], len(source))
            elif source[pos:].strip():
                filled.extend(self.split_span(source, pos, len(source)))

        return filled

    def build_chunks(self, spans: list[Span], source: str, file_path: str) -> list[Chunk]:
        """Turn boundary spans into `Chunk` objects, dropping blank ones."""
        chunks = []
        for start, end in self.fill_gaps(spans, source):
            text = source[start:end]
            if text.strip():
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        first_character_index=start,
                        last_character_index=end,
                        text=text,
                    )
                )
        return chunks

    # ------------------------------------------------------------------
    # Python chunking
    # ------------------------------------------------------------------

    def emit_stmts(
        self,
        stmts: list[ast.stmt],
        spans: list[Span],
        source: str,
        offsets: list[int],
    ) -> None:
        """Group consecutive simple statements into one span; dispatch defs."""
        # Buffer of (start, end) spans for "misc" statements waiting to be
        # flushed as one merged span, so we never recompute node_span twice.
        pending: list[Span] = []

        def flush() -> None:
            if not pending:
                return
            first = pending[0][0]
            last = pending[-1][1]
            if last - first <= self.max_chunk_size:
                spans.append((first, last))
            else:
                spans.extend(self.split_span(source, first, last))
            pending.clear()

        for node in stmts:
            if isinstance(node, self.DEF_NODES):
                flush()
                self.emit_def(node, spans, source, offsets)
                continue

            start, end = node_span(node, offsets, source)

            if end - start > self.max_chunk_size:
                flush()
                spans.extend(self.split_span(source, start, end))
                continue

            if pending and end - pending[0][0] > self.max_chunk_size:
                flush()

            pending.append((start, end))

        flush()

    def emit_def(
        self,
        node: ast.stmt,
        spans: list[Span],
        source: str,
        offsets: list[int],
    ) -> None:
        """Chunk a function/class definition, splitting oversized classes."""
        start, end = node_span(node, offsets, source)
        if end - start <= self.max_chunk_size:
            spans.append((start, end))
            return

        if not (isinstance(node, ast.ClassDef) and node.body):
            spans.extend(self.split_span(source, start, end))
            return

        self._emit_oversized_class(node, spans, source, offsets, start)

    def _emit_oversized_class(
        self,
        node: ast.ClassDef,
        spans: list[Span],
        source: str,
        offsets: list[int],
        start: int,
    ) -> None:
        """Split a class too large for one chunk into header + body spans.

        Glues the class header (signature, docstring, decorators) onto the
        first body span when the combined size still fits, so small classes
        don't get an awkward header-only chunk.
        """
        body_start = node_span(node.body[0], offsets, source)[0]
        spans.extend(self.split_span(source, start, body_start))
        header_span_index = len(spans) - 1

        first_body_span_index = len(spans)
        self.emit_stmts(node.body, spans, source, offsets)

        if len(spans) <= first_body_span_index:
            return  # body produced no spans; nothing to glue

        header_start, header_end = spans[header_span_index]
        body_start_idx, body_end_idx = spans[first_body_span_index]
        if header_end == body_start_idx and body_end_idx - header_start <= self.max_chunk_size:
            spans[header_span_index : first_body_span_index + 1] = [(header_start, body_end_idx)]

    def chunk_python(self, source: str, file_path: str) -> list[Chunk]:
        """Chunk Python source using AST boundaries; falls back to text on syntax errors."""
        try:
            tree = ast.parse(source)
        except SyntaxError as err:
            print(f"SyntaxError in {file_path}, falling back to text: {err}", file=sys.stderr)
            return self.chunk_text(source, file_path)

        offsets = line_offsets(source)
        spans: list[Span] = []
        self.emit_stmts(tree.body, spans, source, offsets)
        return self.build_chunks(spans, source, file_path)

    # ------------------------------------------------------------------
    # Markdown chunking
    # ------------------------------------------------------------------

    def _heading_starts(self, source: str, offsets: list[int]) -> list[int]:
        """Return character indices where top-level headings begin.

        Headings inside fenced code blocks are ignored. A fence only
        closes when its closing marker matches the opening one (```` ``` ````
        vs ``~~~``), so a different fence character nested inside a block
        does not prematurely end it.
        """
        starts = [0]
        fence_char: str | None = None

        for i, line in enumerate(source.split("\n")):
            match = self.FENCE_RE.match(line.lstrip())
            if match:
                if fence_char is None:
                    fence_char = match.group(1)
                elif match.group(1) == fence_char:
                    fence_char = None
                continue

            if fence_char is None and self.HEADING_RE.match(line) and offsets[i] > 0:
                starts.append(offsets[i])

        starts.append(len(source))
        return starts

    def chunk_markdown(self, source: str, file_path: str) -> list[Chunk]:
        """Chunk Markdown by heading sections, merging small adjacent ones."""
        offsets = line_offsets(source)
        starts = self._heading_starts(source, offsets)

        spans: list[Span] = []
        cur_start = 0
        cur_end = 0

        for i in range(len(starts) - 1):
            sec_start, sec_end = starts[i], starts[i + 1]

            if sec_end - cur_start <= self.max_chunk_size:
                cur_end = sec_end
                continue

            if cur_end > cur_start:
                spans.append((cur_start, cur_end))

            cur_start, cur_end = sec_start, sec_end
            if sec_end - sec_start > self.max_chunk_size:
                spans.extend(self.split_span(source, sec_start, sec_end))
                cur_start = cur_end = sec_end

        if cur_end > cur_start:
            spans.append((cur_start, cur_end))

        return self.build_chunks(spans, source, file_path)

    # ------------------------------------------------------------------
    # Plain text / dispatch
    # ------------------------------------------------------------------

    def chunk_text(self, source: str, file_path: str) -> list[Chunk]:
        """Chunk arbitrary text by size, preferring blank-line boundaries."""
        spans = self.split_span(source, 0, len(source))
        return self.build_chunks(spans, source, file_path)

    def chunk_file(self, path: Path, repo_root: Path) -> list[Chunk]:
        """Read a file from disk and dispatch it to the right chunk strategy."""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as err:
            print(f"Skipping {path}: {err}", file=sys.stderr)
            return []

        rel_path = str(path.relative_to(repo_root))

        if path.suffix == ".py":
            return self.chunk_python(content, rel_path)
        if path.suffix in (".md", ".markdown"):
            return self.chunk_markdown(content, rel_path)
        return self.chunk_text(content, rel_path)
