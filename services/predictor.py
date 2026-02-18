import joblib
import numpy as np
import pandas as pd
import os
import json

_model = None

FEATURES = [
    "total_lines", "num_functions", "num_classes", "num_imports",
    "avg_func_length", "max_func_length", "avg_complexity",
    "max_complexity", "high_complexity_fns", "maintainability",
    "lines_per_function", "num_nodes", "num_edges", "avg_degree",
    "max_degree", "graph_density", "avg_shortest_path",
    "clustering_coeff", "deep_nesting_count"
]

_threshold = 0.5

def get_model():
    global _model, _threshold
    if _model is None:
        model_path     = os.path.join(os.path.dirname(__file__), "../ml/bug_model.pkl")
        config_path    = os.path.join(os.path.dirname(__file__), "../ml/model_config.json")
        _model         = joblib.load(model_path)

        if os.path.exists(config_path):
            with open(config_path) as f:
                _threshold = json.load(f).get("threshold", 0.5)

    return _model, _threshold

def predict_bug_probability(metrics: dict, ast_info: dict, graph_features: dict = None) -> dict:
    functions    = ast_info.get("functions", [])
    func_lengths = [f.get("line_count", 0) for f in functions]
    gf           = graph_features or {}

    data = pd.DataFrame([{
        "total_lines":         metrics.get("total_lines", 0),
        "num_functions":       len(functions),
        "num_classes":         len(ast_info.get("classes", [])),
        "num_imports":         len(ast_info.get("imports", [])),
        "avg_func_length":     np.mean(func_lengths) if func_lengths else 0,
        "max_func_length":     np.max(func_lengths)  if func_lengths else 0,
        "avg_complexity":      metrics.get("avg_complexity", 0),
        "max_complexity":      metrics.get("max_complexity", 0),
        "high_complexity_fns": len(metrics.get("complex_functions", [])),
        "maintainability":     metrics.get("maintainability_index", 100),
        "lines_per_function":  metrics.get("total_lines", 0) / max(len(functions), 1),
        "num_nodes":           gf.get("num_nodes", 0),
        "num_edges":           gf.get("num_edges", 0),
        "avg_degree":          gf.get("avg_degree", 0),
        "max_degree":          gf.get("max_degree", 0),
        "graph_density":       gf.get("density", 0),
        "avg_shortest_path":   gf.get("avg_shortest_path", 0),
        "clustering_coeff":    gf.get("clustering_coefficient", 0),
        "deep_nesting_count":  gf.get("deep_nesting_count", 0),
    }])

    model, threshold = get_model()
    prob             = model.predict_proba(data)[0][1]

    return {
        "bug_probability": round(float(prob), 3),
        "is_likely_buggy": bool(prob >= threshold),   # uses tuned threshold
        "confidence":      "high" if prob > 0.75 or prob < 0.25 else "medium"
    }