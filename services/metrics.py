from radon.complexity import cc_visit
from radon.metrics import mi_visit
from dataclasses import dataclass

@dataclass
class FileMetrics:
    path: str
    total_lines: int
    avg_complexity: float
    max_complexity: float
    maintainability_index: float
    complex_functions: list
    risk_score: float

def compute_complexity(content: str, path: str) -> FileMetrics:
    """Compute cyclomatic complexity and maintainability index"""
    try:
        complexity_results = cc_visit(content)
    except Exception:
        complexity_results = []

    try:
        mi_score = mi_visit(content, multi=True)
    except Exception:
        mi_score = 100.0  # default to perfect if it fails

    complexities = [r.complexity for r in complexity_results]
    avg_complexity = sum(complexities) / len(complexities) if complexities else 0
    max_complexity = max(complexities) if complexities else 0

    # Flag functions with complexity > 10 (industry standard threshold)
    complex_functions = [
        {"name": r.name, "complexity": r.complexity, "line": r.lineno}
        for r in complexity_results if r.complexity > 10
    ]

    risk_score = compute_risk_score(avg_complexity, max_complexity, mi_score)

    return FileMetrics(
        path=path,
        total_lines=content.count("\n") + 1,
        avg_complexity=round(avg_complexity, 2),
        max_complexity=round(max_complexity, 2),
        maintainability_index=round(mi_score, 2),
        complex_functions=complex_functions,
        risk_score=risk_score
    )

def compute_risk_score(avg_complexity: float, max_complexity: float, mi: float) -> float:
    """
    Score from 0 (safe) to 100 (very risky)
    Weighted blend of complexity and maintainability
    """
    # Normalize complexity: cap at 30 for scoring
    complexity_score = min((avg_complexity / 30) * 60, 60)

    # Maintainability index is 0-100, higher = better
    # Invert it so higher MI = lower risk contribution
    mi_score = max(0, (100 - mi) / 100 * 40)

    return round(complexity_score + mi_score, 1)