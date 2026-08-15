import ast

tree = ast.parse(f.readlines())
print(type(tree).__name__)
print(tree.body)
