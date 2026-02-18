import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib
import ast
import os
from radon.complexity import cc_visit
from radon.metrics import mi_visit

# ── Feature extraction (same logic as your existing services) ──────────────
def extract_features(content: str) -> dict | None:
    """Extract numeric features from raw Python source code"""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    # AST features
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes   = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    imports   = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]

    func_lengths = [
        (n.end_lineno or n.lineno) - n.lineno for n in functions
    ]

    # Complexity features
    try:
        cc_results   = cc_visit(content)
        complexities = [r.complexity for r in cc_results]
        avg_cc       = np.mean(complexities) if complexities else 0
        max_cc       = np.max(complexities)  if complexities else 0
        high_cc      = sum(1 for c in complexities if c > 10)
    except Exception:
        avg_cc = max_cc = high_cc = 0

    # Maintainability
    try:
        mi = mi_visit(content, multi=True)
    except Exception:
        mi = 100.0

    total_lines = content.count("\n") + 1

    return {
        "total_lines":       total_lines,
        "num_functions":     len(functions),
        "num_classes":       len(classes),
        "num_imports":       len(imports),
        "avg_func_length":   np.mean(func_lengths) if func_lengths else 0,
        "max_func_length":   np.max(func_lengths)  if func_lengths else 0,
        "avg_complexity":    avg_cc,
        "max_complexity":    max_cc,
        "high_complexity_fns": high_cc,
        "maintainability":   mi,
        "lines_per_function": total_lines / max(len(functions), 1),
    }


# ── Dataset: synthetic but realistic ──────────────────────────────────────
def generate_dataset(n_samples: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    rows = []

    for _ in range(n_samples):
        is_buggy = np.random.rand() < 0.4

        if is_buggy:
            total_lines         = np.random.randint(100, 1500)
            num_functions       = np.random.randint(5, 40)
            avg_complexity      = np.random.uniform(8, 30)
            max_complexity      = avg_complexity + np.random.uniform(5, 20)
            maintainability     = np.random.uniform(10, 55)
            high_complexity_fns = np.random.randint(2, 15)
            # Graph features for buggy files — highly connected, deep
            num_nodes           = np.random.randint(10, 60)
            num_edges           = np.random.randint(15, 100)
            avg_degree          = np.random.uniform(2.5, 8.0)
            max_degree          = np.random.randint(5, 20)
            density             = np.random.uniform(0.15, 0.6)
            avg_shortest_path   = np.random.uniform(2.5, 6.0)
            clustering_coeff    = np.random.uniform(0.3, 0.9)
            deep_nesting        = np.random.randint(3, 15)
        else:
            total_lines         = np.random.randint(20, 400)
            num_functions       = np.random.randint(1, 15)
            avg_complexity      = np.random.uniform(1, 8)
            max_complexity      = avg_complexity + np.random.uniform(0, 8)
            maintainability     = np.random.uniform(55, 100)
            high_complexity_fns = np.random.randint(0, 2)
            # Graph features for clean files — sparse, shallow
            num_nodes           = np.random.randint(2, 15)
            num_edges           = np.random.randint(1, 20)
            avg_degree          = np.random.uniform(0.5, 2.5)
            max_degree          = np.random.randint(1, 5)
            density             = np.random.uniform(0.0, 0.15)
            avg_shortest_path   = np.random.uniform(1.0, 2.5)
            clustering_coeff    = np.random.uniform(0.0, 0.3)
            deep_nesting        = np.random.randint(0, 3)

        avg_func_length    = total_lines / max(num_functions, 1)
        max_func_length    = avg_func_length * np.random.uniform(1.2, 3.0)
        num_classes        = np.random.randint(0, 6)
        num_imports        = np.random.randint(1, 20)
        lines_per_function = total_lines / max(num_functions, 1)

        rows.append({
            # Original features
            "total_lines":          total_lines,
            "num_functions":        num_functions,
            "num_classes":          num_classes,
            "num_imports":          num_imports,
            "avg_func_length":      avg_func_length,
            "max_func_length":      max_func_length,
            "avg_complexity":       avg_complexity,
            "max_complexity":       max_complexity,
            "high_complexity_fns":  high_complexity_fns,
            "maintainability":      maintainability,
            "lines_per_function":   lines_per_function,
            # New graph features
            "num_nodes":            num_nodes,
            "num_edges":            num_edges,
            "avg_degree":           avg_degree,
            "max_degree":           max_degree,
            "graph_density":        density,
            "avg_shortest_path":    avg_shortest_path,
            "clustering_coeff":     clustering_coeff,
            "deep_nesting_count":   deep_nesting,
            "is_buggy":             int(is_buggy),
        })

    return pd.DataFrame(rows)

# ── Train ──────────────────────────────────────────────────────────────────
def train():
    print("Generating dataset...")
    df = generate_dataset(2000)

    FEATURES = [
    "total_lines", "num_functions", "num_classes", "num_imports",
    "avg_func_length", "max_func_length", "avg_complexity",
    "max_complexity", "high_complexity_fns", "maintainability",
    "lines_per_function",
    # Graph features
    "num_nodes", "num_edges", "avg_degree", "max_degree",
    "graph_density", "avg_shortest_path", "clustering_coeff",
    "deep_nesting_count"
]

    X = df[FEATURES]
    y = df["is_buggy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        ))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Clean", "Buggy"]))

    # Save model
    os.makedirs("ml", exist_ok=True)
    joblib.dump(model, "ml/bug_model.pkl")
    print("\n✅ Model saved to ml/bug_model.pkl")


if __name__ == "__main__":
    train()