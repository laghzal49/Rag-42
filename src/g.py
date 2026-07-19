import ast

src = open("data/raw/vllm-0.10.1/vllm/core/scheduler.py").read()
tree = ast.parse(src)
for node in ast.iter_child_nodes(tree):
    print(
        type(node).__name__,
        getattr(node, "name", ""),
        node.lineno,
        node.end_lineno,
    )
sched = [
    n
    for n in ast.iter_child_nodes(tree)
    if getattr(n, "name", "") == "Scheduler"
][0]
for node in sched.body:
    print(
        type(node).__name__,
        getattr(node, "name", ""),
        node.lineno,
        node.end_lineno,
    )
