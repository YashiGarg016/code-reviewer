import re
from dataclasses import dataclass

@dataclass
class JSFileMetrics:
    path: str
    total_lines: int
    avg_complexity: float
    max_complexity: float
    maintainability_index: float
    complex_functions: list
    risk_score: float

def compute_js_complexity(content: str, path: str) -> JSFileMetrics:
    """
    Compute cyclomatic complexity for JS/TS by counting
    branching keywords — same underlying theory as radon for Python.
    """
    lines       = content.split("\n")
    total_lines = len(lines)
    functions   = _split_into_functions(content)

    complexities      = []
    complex_functions = []

    for fn_name, fn_body, start_line in functions:
        complexity = _cyclomatic_complexity(fn_body)
        complexities.append(complexity)
        if complexity > 10:
            complex_functions.append({
                "name":       fn_name,
                "complexity": complexity,
                "line":       start_line
            })

    avg_complexity = sum(complexities) / len(complexities) if complexities else 0
    max_complexity = max(complexities) if complexities else 0

    # Maintainability: penalize long files and high complexity
    mi = max(0, 100 - (total_lines / 10) - (avg_complexity * 3))
    mi = min(100, mi)

    risk_score = _compute_risk(avg_complexity, max_complexity, mi)

    return JSFileMetrics(
        path=path,
        total_lines=total_lines,
        avg_complexity=round(avg_complexity, 2),
        max_complexity=round(max_complexity, 2),
        maintainability_index=round(mi, 2),
        complex_functions=complex_functions,
        risk_score=risk_score
    )


def _cyclomatic_complexity(code: str) -> int:
    """
    Count branching points — each adds 1 to complexity.
    Starts at 1 (the function itself is one path).
    """
    branch_keywords = [
        r"\bif\b", r"\belse\s+if\b", r"\bfor\b", r"\bwhile\b",
        r"\bdo\b",  r"\bcase\b",      r"\bcatch\b", r"\b\?\b",
        r"&&",      r"\|\|",          r"\?\?",       r"\bswitch\b"
    ]
    complexity = 1
    for pattern in branch_keywords:
        complexity += len(re.findall(pattern, code))
    return complexity


def _split_into_functions(content: str) -> list:
    """Split content into (name, body, start_line) tuples"""
    functions = []
    lines     = content.split("\n")

    fn_pattern = re.compile(
        r"(?:async\s+)?function\s+(\w+)|"
        r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\(|\w+\s*=>)|"
        r"(?:public|private|protected|static)?\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{"
    )

    for i, line in enumerate(lines):
        match = fn_pattern.search(line)
        if match:
            name = next((g for g in match.groups() if g), "anonymous")
            # Grab next 100 lines as function body
            body = "\n".join(lines[i:min(i + 100, len(lines))])
            functions.append((name, body, i + 1))

    # If no functions found, treat whole file as one unit
    if not functions:
        functions.append(("__module__", content, 1))

    return functions


def _compute_risk(avg_cc: float, max_cc: float, mi: float) -> float:
    complexity_score = min((avg_cc / 30) * 60, 60)
    mi_score         = max(0, (100 - mi) / 100 * 40)
    return round(complexity_score + mi_score, 1)