import ast
from dataclasses import dataclass

@dataclass
class FunctionInfo:
    name: str
    line_start: int
    line_end: int
    line_count: int
    arg_count: int
    is_complex: bool  # flags functions worth warning about

def parse_file(content: str) -> dict:
    """Parse a Python file and extract structural info"""
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"error": str(e), "functions": [], "classes": [], "imports": []}

    functions = []
    classes = []
    imports = []

    for node in ast.walk(tree):
        # Extract functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            line_count = (node.end_lineno or node.lineno) - node.lineno
            functions.append(FunctionInfo(
                name=node.name,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                line_count=line_count,
                arg_count=len(node.args.args),
                is_complex=line_count > 50 or len(node.args.args) > 5
            ).__dict__)

        # Extract classes
        elif isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "method_count": sum(1 for n in ast.walk(node) if isinstance(n, ast.FunctionDef))
            })

        # Extract imports
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    return {
        "functions": functions,
        "classes": classes,
        "imports": list(set(imports)),  # deduplicate
        "total_lines": content.count("\n") + 1
    }