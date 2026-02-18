import re
from dataclasses import dataclass

@dataclass
class JSFileInfo:
    functions: list
    classes: list
    imports: list
    total_lines: int
    error: str = None

def parse_js_file(content: str) -> dict:
    """
    Parse JavaScript/TypeScript using regex-based AST extraction.
    Covers functions, classes, imports reliably without heavy dependencies.
    """
    try:
        lines = content.split("\n")
        total_lines = len(lines)

        functions = _extract_functions(content)
        classes   = _extract_classes(content)
        imports   = _extract_imports(content)

        return {
            "functions": functions,
            "classes":   classes,
            "imports":   imports,
            "total_lines": total_lines,
            "error": None
        }
    except Exception as e:
        return {
            "functions": [], "classes": [], "imports": [],
            "total_lines": 0, "error": str(e)
        }


def _extract_functions(content: str) -> list:
    functions = []
    lines     = content.split("\n")

    patterns = [
        # function foo() / async function foo()
        r"(?:async\s+)?function\s+(\w+)\s*\(",
        # const foo = () => / const foo = async () =>
        r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(",
        # const foo = () => (arrow shorthand)
        r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\w+|\(.*?\))\s*=>",
        # class method: foo() { / async foo() {
        r"^\s+(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{",
        # TypeScript: public/private/protected methods
        r"(?:public|private|protected|static)\s+(?:async\s+)?(\w+)\s*\(",
    ]

    for i, line in enumerate(lines):
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                name = match.group(1)
                # Skip keywords mistaken as function names
                if name in ("if", "for", "while", "switch", "catch", "constructor"):
                    continue
                # Estimate function length by looking for closing brace
                end_line = _find_function_end(lines, i)
                line_count = end_line - i

                functions.append({
                    "name":       name,
                    "line_start": i + 1,
                    "line_end":   end_line + 1,
                    "line_count": line_count,
                    "arg_count":  line.count(",") + 1 if "(" in line else 0,
                    "is_complex": line_count > 50
                })
                break  # avoid double-matching same line

    return functions


def _find_function_end(lines: list, start: int) -> int:
    """Find closing brace of a function by tracking brace depth"""
    depth = 0
    for i in range(start, min(start + 200, len(lines))):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth <= 0 and i > start:
            return i
    return min(start + 50, len(lines) - 1)


def _extract_classes(content: str) -> list:
    classes  = []
    lines    = content.split("\n")
    pattern  = r"class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{"

    for i, line in enumerate(lines):
        match = re.search(pattern, line)
        if match:
            name         = match.group(1)
            method_count = sum(
                1 for l in lines[i:i+200]
                if re.search(r"^\s+(?:async\s+)?(\w+)\s*\(", l)
            )
            classes.append({
                "name":         name,
                "line":         i + 1,
                "method_count": method_count
            })

    return classes


def _extract_imports(content: str) -> list:
    imports  = []
    patterns = [
        r"import\s+.*?\s+from\s+['\"](.+?)['\"]",   # ES6 import
        r"require\s*\(\s*['\"](.+?)['\"]\s*\)",       # CommonJS require
        r"import\s*\(\s*['\"](.+?)['\"]\s*\)",        # Dynamic import
    ]
    for pattern in patterns:
        imports.extend(re.findall(pattern, content))

    return list(set(imports))