import ast

src = """
class dt():
    def hey():
        print("hello")

class f:
    def t():
        f = 2
"""


def line_offsets(source: str):
    offs = [0]
    for line in source.split("\n"):
        offs.append(offs[-1] + len(line) + 1)
    return offs


def node_span(node, offs):
    decor = getattr(node, "decorator_list", [])
    if decor:
        start_line = decor[0].lineno
    else:
        start_line = node.lineno
    last = offs[node.end_lineno - 1] + node.end_col_offset
    start = offs[start_line - 1]
    return start, last


tree: ast.AST = ast.parse(src)
es = line_offsets(src)
for node in ast.iter_child_nodes(tree):
    if isinstance(node, ast.ClassDef):
        for smt in node.body:
            for g in smt.body:
                print(type(smt).__name__, getattr(smt, "name", ""))

            start, end = node_span(smt, es)
            if end - start < 50:
                print(src[start:end])
                print("--------------------------------")
