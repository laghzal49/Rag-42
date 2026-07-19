import ast
import sys
from pathlib import Path

from pydantic import BaseModel

# ---- the chunk itself ----
# Could be a pydantic model (subject wants pydantic for data exchanged
# between stages) or dataclass — your call, but chunks flow from indexer
# to retriever, so pydantic fits the requirement.


class Chunk(BaseModel):
    file_path: str  # relative, exact corpus convention: data/raw/vllm-0.10.1/
    first_character_index: int
    last_character_index: int
    text: str  # == source[first:last], always


# ---- public API, one per strategy ----


def chunk_python(
    source: str, file_path: str, max_chunk_size: int
) -> list[Chunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(e)
    for node in ast.iter_child_nodes(tree):
        ...


def chunk_markdown(
    source: str, file_path: str, max_chunk_size: int
) -> list[Chunk]: ...


# ---- dispatcher the indexer calls ----


def chunk_file(
    path: Path, repo_root: Path, max_chunk_size: int
) -> list[Chunk]:
    ...
    # reads the file, computes rel path, routes .py → chunk_python,
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        print(f"skipping {path}: {e}", file=sys.stderr)
        return []
    rel_path = str(path.relative_to(repo_root))
    # .md/.txt/etc → chunk_markdown, returns []or fallback for others
    if path.suffix == ".py":
        chunk = chunk_python(content, rel_path, max_chunk_size)
    elif path.suffix == ".md":
        chunk = chunk_markdown(content, rel_path, max_chunk_size)
    else:
        return []
    return chunk
