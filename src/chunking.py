import ast
import re
import sys
from pathlib import Path

from pydantic import BaseModel

DEF_NODES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
HEADING_RE = re.compile(r"^#{1,6}\s")
FENCE_RE = re.compile(r"^(```|~~~)")

Span = tuple[int, int]


class Chunk(BaseModel):
    """One retrievable piece of a file, with exact character indices."""

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str


def line_offsets(source: str) -> list[int]:
    """Starting character index of every line in source."""
    offs = [0]
    for line in source.split("\n"):
        offs.append(offs[-1] + len(line) + 1)
    return offs


def node_span(node: ast.stmt, offs: list[int], source: str) -> Span:
    """Character span of a statement, decorators included."""
    decorators = getattr(node, "decorator_list", [])
    start_line = decorators[0].lineno if decorators else node.lineno
    start = offs[start_line - 1]
    line_start = offs[node.end_lineno - 1]
    line = source[line_start : offs[node.end_lineno] - 1]
    col = node.end_col_offset
    if not line.isascii():
        # ast column offsets count utf-8 bytes, not characters
        col = len(line.encode()[:col].decode("utf-8", "ignore"))
    return start, line_start + col


def split_span(
    source: str, start: int, end: int, max_chunk_size: int
) -> list[Span]:
    """Cut [start, end) into pieces <= max_chunk_size, breaking at a
    blank line if possible, else a line end, else a hard cut."""
    spans = []
    pos = start
    while end - pos > max_chunk_size:
        window = source[pos : pos + max_chunk_size]
        cut = window.rfind("\n\n")
        if cut != -1:
            cut += 2
        else:
            cut = window.rfind("\n") + 1
        if cut <= 0:
            cut = max_chunk_size
        spans.append((pos, pos + cut))
        pos += cut
    spans.append((pos, end))
    return spans


def fill_gaps(
    spans: list[Span], source: str, max_chunk_size: int
) -> list[Span]:
    """Stretch spans over the text between them (comments, blank
    lines) so the whole file stays covered."""
    filled: list[Span] = []
    pos = 0
    for start, end in spans:
        if start > pos:
            if end - pos <= max_chunk_size:
                start = pos
            elif filled and start - filled[-1][0] <= max_chunk_size:
                filled[-1] = (filled[-1][0], start)
            elif source[pos:start].strip():
                filled.extend(
                    split_span(source, pos, start, max_chunk_size)
                )
        filled.append((start, end))
        pos = end
    if pos < len(source):
        if filled and len(source) - filled[-1][0] <= max_chunk_size:
            filled[-1] = (filled[-1][0], len(source))
        elif source[pos:].strip():
            filled.extend(
                split_span(source, pos, len(source), max_chunk_size)
            )
    return filled


def build_chunks(
    spans: list[Span],
    source: str,
    file_path: str,
    max_chunk_size: int,
) -> list[Chunk]:
    """Fill the gaps between spans and drop whitespace-only pieces."""
    chunks = []
    for start, end in fill_gaps(spans, source, max_chunk_size):
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


def emit_stmts(
    stmts: list[ast.stmt],
    spans: list[Span],
    source: str,
    offs: list[int],
    max_chunk_size: int,
) -> None:
    """One chunk per def/class, consecutive simple statements grouped
    together up to max_chunk_size."""
    misc: list[ast.stmt] = []

    def flush() -> None:
        if misc:
            first, _ = node_span(misc[0], offs, source)
            _, last = node_span(misc[-1], offs, source)
            spans.append((first, last))
            misc.clear()

    for node in stmts:
        if isinstance(node, DEF_NODES):
            flush()
            emit_def(node, spans, source, offs, max_chunk_size)
            continue
        start, end = node_span(node, offs, source)
        if end - start > max_chunk_size:
            flush()
            spans.extend(
                split_span(source, start, end, max_chunk_size)
            )
            continue
        if (
            misc
            and end - node_span(misc[0], offs, source)[0]
            > max_chunk_size
        ):
            flush()
        misc.append(node)
    flush()


def emit_def(
    node: ast.stmt,
    spans: list[Span],
    source: str,
    offs: list[int],
    max_chunk_size: int,
) -> None:
    """Emit a def/class whole if it fits; an oversized class becomes a
    header chunk plus its body chunked like a module, an oversized
    function is split at blank lines."""
    start, end = node_span(node, offs, source)
    if end - start <= max_chunk_size:
        spans.append((start, end))
        return
    if isinstance(node, ast.ClassDef) and node.body:
        body_start = node_span(node.body[0], offs, source)[0]
        spans.extend(
            split_span(source, start, body_start, max_chunk_size)
        )
        first_body = len(spans)
        emit_stmts(node.body, spans, source, offs, max_chunk_size)
        # glue the class line onto the first body chunk when both fit
        if len(spans) > first_body:
            h_start, h_end = spans[first_body - 1]
            b_start, b_end = spans[first_body]
            if h_end == b_start and b_end - h_start <= max_chunk_size:
                spans[first_body - 1 : first_body + 1] = [
                    (h_start, b_end)
                ]
    else:
        spans.extend(split_span(source, start, end, max_chunk_size))


def chunk_python(
    source: str, file_path: str, max_chunk_size: int
) -> list[Chunk]:
    """AST chunking of a python file; falls back to plain text when
    the file does not parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(
            f"SyntaxError in {file_path}, plain-text fallback: {e}",
            file=sys.stderr,
        )
        return chunk_text(source, file_path, max_chunk_size)
    offs = line_offsets(source)
    spans: list[Span] = []
    emit_stmts(tree.body, spans, source, offs, max_chunk_size)
    return build_chunks(spans, source, file_path, max_chunk_size)


def chunk_markdown(
    source: str, file_path: str, max_chunk_size: int
) -> list[Chunk]:
    """Split on headings (outside code fences), merge consecutive
    sections up to max_chunk_size, split oversized ones."""
    offs = line_offsets(source)
    starts = [0]
    in_fence = False
    for i, line in enumerate(source.split("\n")):
        if FENCE_RE.match(line.lstrip()):
            in_fence = not in_fence
        elif not in_fence and HEADING_RE.match(line) and offs[i] > 0:
            starts.append(offs[i])
    starts.append(len(source))

    spans: list[Span] = []
    cur_start = 0
    cur_end = 0
    for i in range(len(starts) - 1):
        sec_start, sec_end = starts[i], starts[i + 1]
        if sec_end - cur_start <= max_chunk_size:
            cur_end = sec_end
            continue
        if cur_end > cur_start:
            spans.append((cur_start, cur_end))
        cur_start, cur_end = sec_start, sec_end
        if sec_end - sec_start > max_chunk_size:
            spans.extend(
                split_span(source, sec_start, sec_end, max_chunk_size)
            )
            cur_start = cur_end = sec_end
    if cur_end > cur_start:
        spans.append((cur_start, cur_end))
    return build_chunks(spans, source, file_path, max_chunk_size)


def chunk_text(
    source: str, file_path: str, max_chunk_size: int
) -> list[Chunk]:
    """Plain-text chunking at blank-line / line boundaries."""
    spans = split_span(source, 0, len(source), max_chunk_size)
    return build_chunks(spans, source, file_path, max_chunk_size)


# ---- dispatcher the indexer calls ----


def chunk_file(
    path: Path, repo_root: Path, max_chunk_size: int
) -> list[Chunk]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        print(f"skipping {path}: {e}", file=sys.stderr)
        return []
    rel_path = str(path.relative_to(repo_root))
    if path.suffix == ".py":
        return chunk_python(content, rel_path, max_chunk_size)
    if path.suffix in (".md", ".markdown"):
        return chunk_markdown(content, rel_path, max_chunk_size)
    if path.suffix in (".txt", ".rst"):
        return chunk_text(content, rel_path, max_chunk_size)
    return []
