import ast

with open("main.py") as f:
    tree = ast.parse(f.readlines())
print(type(tree).__name__)
