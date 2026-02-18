"""
Train bug prediction model on real mined dataset.
Replaces the synthetic training data with actual labeled code files.
Improvements over v1:
- Class imbalance correction via sample weights
- Lower decision threshold to improve bug recall
- Better regularization to reduce overfitting
- Saves threshold to model_config.json for use at inference time
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
import joblib

FEATURES = [
    "total_lines", "num_functions", "num_classes", "num_imports",
    "avg_func_length", "max_func_length", "avg_complexity",
    "max_complexity", "high_complexity_fns", "maintainability",
    "lines_per_function", "num_nodes", "num_edges", "avg_degree",
    "max_degree", "graph_density", "avg_shortest_path",
    "clustering_coeff", "deep_nesting_count"
]


def train():
    csv_path = "ml/real_dataset.csv"

    if not os.path.exists(csv_path):
        print("✗ real_dataset.csv not found. Run collect_data.py first.")
        return

    print("Loading real dataset...")
    df = pd.read_csv(csv_path)

    print(f"\n📊 Dataset loaded:")
    print(f"  Total samples: {len(df)}")
    print(f"  Buggy:         {df['is_buggy'].sum()}")
    print(f"  Clean:         {(df['is_buggy'] == 0).sum()}")
    print(f"  Buggy ratio:   {df['is_buggy'].mean():.1%}")

    # Drop rows with missing features
    df = df.dropna(subset=FEATURES)
    print(f"  After cleaning: {len(df)} samples")

    if len(df) < 50:
        print("✗ Not enough samples to train. Try collecting more data.")
        return

    X = df[FEATURES]
    y = df["is_buggy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fix class imbalance — give buggy samples more weight during training
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=400,
            max_depth=3,            # shallower trees = less overfitting
            learning_rate=0.03,     # slower learning = better generalization
            subsample=0.7,          # use 70% of samples per tree
            min_samples_leaf=15,    # leaf needs 15 samples minimum
            max_features=0.8,       # use 80% of features per split
            random_state=42
        ))
    ])

    print("\nTraining on real data...")
    model.fit(X_train, y_train, clf__sample_weight=sample_weights)

    # ── Default threshold evaluation ──────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n📈 Classification Report (default threshold=0.5):")
    print(classification_report(y_test, y_pred, target_names=["Clean", "Buggy"]))
    print(f"ROC-AUC Score: {roc_auc:.3f}")

    # ── Lowered threshold — improves bug recall ───────────────────────────
    # Lowering from 0.5 → 0.4 catches more buggy files at cost of more false positives
    # For a code review tool, missing bugs (false negatives) is worse than over-flagging
    threshold = 0.40
    y_pred_adjusted = (y_prob >= threshold).astype(int)

    print(f"\n📈 Adjusted Report (threshold={threshold} — better bug recall):")
    print(classification_report(
        y_test, y_pred_adjusted, target_names=["Clean", "Buggy"]
    ))

    # ── 5-Fold Cross Validation ───────────────────────────────────────────
    print("\n🔁 5-Fold Cross Validation:")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"  AUC per fold: {[round(float(s), 3) for s in cv_auc]}")
    print(f"  Mean AUC:     {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    # ── Feature Importance ────────────────────────────────────────────────
    importances = model.named_steps["clf"].feature_importances_
    feat_imp    = sorted(
        zip(FEATURES, importances), key=lambda x: x[1], reverse=True
    )
    print("\n🔑 Top 10 Most Important Features:")
    for feat, imp in feat_imp[:10]:
        bar = "█" * int(imp * 200)
        print(f"  {feat:<25} {bar} {imp:.4f}")

    # ── Save model + config ───────────────────────────────────────────────
    os.makedirs("ml", exist_ok=True)

    # Save threshold so predictor.py uses the same value at inference time
    config = {
        "threshold":      threshold,
        "roc_auc":        round(float(roc_auc), 3),
        "cv_mean_auc":    round(float(cv_auc.mean()), 3),
        "cv_std_auc":     round(float(cv_auc.std()), 3),
        "total_samples":  len(df),
        "buggy_samples":  int(df["is_buggy"].sum()),
        "clean_samples":  int((df["is_buggy"] == 0).sum()),
        "features":       FEATURES
    }

    with open("ml/model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    joblib.dump(model, "ml/bug_model.pkl")

    print("\n✅ Model saved to ml/bug_model.pkl")
    print("✅ Config saved to ml/model_config.json")
    print("   Restart uvicorn to load the new model.")


if __name__ == "__main__":
    train()