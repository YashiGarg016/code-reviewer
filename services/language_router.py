from services.parser import parse_file
from services.metrics import compute_complexity
from services.security import run_bandit
from services.js_parser import parse_js_file
from services.js_metrics import compute_js_complexity
from services.js_security import run_js_security
from services.graph_builder import extract_graph_features

SUPPORTED_EXTENSIONS = {
    ".py":  "python",
    ".js":  "javascript",
    ".ts":  "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
}

def detect_language(path: str) -> str | None:
    """Return language string or None if unsupported"""
    for ext, lang in SUPPORTED_EXTENSIONS.items():
        if path.endswith(ext):
            return lang
    return None

def analyze_file(content: str, path: str) -> dict:
    """
    Route file to correct parser/metrics/security services
    based on detected language. Returns unified result shape.
    """
    language = detect_language(path)

    if language == "python":
        ast_info  = parse_file(content)
        metrics   = compute_complexity(content, path)
        security  = run_bandit(content, path)

    elif language in ("javascript", "typescript"):
        ast_info  = parse_js_file(content)
        metrics   = compute_js_complexity(content, path)
        security  = run_js_security(content, path)

    else:
        return None  # unsupported

    # Graph analysis works on any language via regex
    graph = extract_graph_features(content)

    return {
        "language": language,
        "ast":      ast_info,
        "metrics":  metrics.__dict__ if hasattr(metrics, "__dict__") else metrics,
        "security": security,
        "graph": {
            "num_nodes":             graph.num_nodes,
            "num_edges":             graph.num_edges,
            "avg_degree":            graph.avg_degree,
            "most_connected_node":   graph.most_connected_node,
            "deep_nesting_count":    graph.deep_nesting_count,
            "clustering_coefficient": graph.clustering_coefficient,
        }
    }