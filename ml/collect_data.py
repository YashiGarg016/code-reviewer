"""
Real bug dataset collector using Mining Software Repositories (MSR) methodology.

Strategy:
- Clone popular Python repos
- Find commits with bug-fix keywords in message
- Label files changed in those commits as BUGGY (pre-fix state)
- Label stable files (untouched for 10+ commits) as CLEAN
- Extract 19 features from real source code
- Save as CSV for training
"""

import os
import sys
import csv
import ast
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from git import Repo, GitCommandError
from pathlib import Path

# Add backend root to path so we can import our services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.parser import parse_file
from services.metrics import compute_complexity
from services.graph_builder import extract_graph_features

# ── Config ────────────────────────────────────────────────────────────────────

REPOS = [
    "https://github.com/pallets/flask",
    "https://github.com/psf/requests",
    "https://github.com/pytest-dev/pytest",
    "https://github.com/tornadoweb/tornado",
    "https://github.com/sqlalchemy/sqlalchemy",
    "https://github.com/celery/celery",
    "https://github.com/scrapy/scrapy",
    "https://github.com/encode/httpx",
    "https://github.com/aio-libs/aiohttp",
    "https://github.com/pallets/click",
    "https://github.com/bottlepy/bottle",
    "https://github.com/falconry/falcon",
    "https://github.com/hugapi/hug",
    "https://github.com/sanic-org/sanic",
    "https://github.com/tiangolo/fastapi",
    "https://github.com/paramiko/paramiko",
    "https://github.com/fabric/fabric",
    "https://github.com/pypa/pip",
    "https://github.com/cookiecutter/cookiecutter",
    "https://github.com/httpie/cli",
]

CLONE_DIR       = Path("ml/repos")          # where repos are cloned
OUTPUT_CSV      = Path("ml/real_dataset.csv")
MAX_COMMITS     = 1000                        # commits to scan per repo
MIN_CLEAN_AGE   = 15                         # file must be untouched for N commits to be "clean"

BUG_KEYWORDS = [
    "fix bug", "bugfix", "bug fix", "fixes bug",
    "fix issue", "fixes issue", "fix error",
    "fix crash", "fix exception", "fix vulnerability",
    "fix security", "patch", "hotfix", "resolve issue",
]

# ── Feature Extraction ────────────────────────────────────────────────────────

def extract_features_from_content(content: str, path: str) -> dict | None:
    """Run full feature extraction pipeline on raw file content"""
    try:
        ast_info     = parse_file(content)
        metrics      = compute_complexity(content, path)
        graph        = extract_graph_features(content)

        if ast_info.get("error"):
            return None  # skip unparseable files

        functions    = ast_info.get("functions", [])
        func_lengths = [f.get("line_count", 0) for f in functions]

        return {
            # Metrics features
            "total_lines":         metrics.total_lines,
            "num_functions":       len(functions),
            "num_classes":         len(ast_info.get("classes", [])),
            "num_imports":         len(ast_info.get("imports", [])),
            "avg_func_length":     np.mean(func_lengths) if func_lengths else 0,
            "max_func_length":     np.max(func_lengths)  if func_lengths else 0,
            "avg_complexity":      metrics.avg_complexity,
            "max_complexity":      metrics.max_complexity,
            "high_complexity_fns": len(metrics.complex_functions),
            "maintainability":     metrics.maintainability_index,
            "lines_per_function":  metrics.total_lines / max(len(functions), 1),
            # Graph features
            "num_nodes":           graph.num_nodes,
            "num_edges":           graph.num_edges,
            "avg_degree":          graph.avg_degree,
            "max_degree":          graph.max_degree,
            "graph_density":       graph.density,
            "avg_shortest_path":   graph.avg_shortest_path,
            "clustering_coeff":    graph.clustering_coefficient,
            "deep_nesting_count":  graph.deep_nesting_count,
        }
    except Exception:
        return None


# ── Git Mining ────────────────────────────────────────────────────────────────

def is_bug_fix_commit(message: str) -> bool:
    """Check if commit message indicates a bug fix"""
    msg = message.lower()
    return any(keyword in msg for keyword in BUG_KEYWORDS)


def get_file_content_at_commit(repo: Repo, commit, path: str) -> str | None:
    """Get file content at a specific commit"""
    try:
        blob = commit.tree / path
        return blob.data_stream.read().decode("utf-8", errors="replace")
    except (KeyError, AttributeError):
        return None


def get_stable_files(repo: Repo, commits: list) -> set:
    """
    Find files that haven't been touched in MIN_CLEAN_AGE commits.
    These are our "clean" examples.
    """
    recently_touched = set()

    # Collect all files touched in recent commits
    for commit in commits[:MIN_CLEAN_AGE]:
        try:
            for file in commit.stats.files.keys():
                if file.endswith(".py"):
                    recently_touched.add(file)
        except Exception:
            continue

    # All py files in HEAD
    all_py_files = set()
    try:
        for blob in repo.head.commit.tree.traverse():
            if hasattr(blob, "path") and blob.path.endswith(".py"):
                all_py_files.add(blob.path)
    except Exception:
        pass

    return all_py_files - recently_touched


def mine_repo(repo_url: str, samples: list) -> int:
    """
    Mine a single repo for buggy and clean file samples.
    Returns number of samples collected.
    """
    repo_name  = repo_url.split("/")[-1]
    clone_path = CLONE_DIR / repo_name
    collected  = 0

    # Clone if not already present
    if not clone_path.exists():
        print(f"  Cloning {repo_name}...")
        try:
            Repo.clone_from(repo_url, clone_path, depth=MAX_COMMITS + 50)
        except GitCommandError as e:
            print(f"  ✗ Failed to clone {repo_name}: {e}")
            return 0
    else:
        print(f"  Using cached {repo_name}")

    try:
        repo    = Repo(clone_path)
        commits = list(repo.iter_commits("HEAD", max_count=MAX_COMMITS))
    except Exception as e:
        print(f"  ✗ Failed to read {repo_name}: {e}")
        return 0

    # ── Collect BUGGY samples from bug-fix commits ────────────────────────
    print(f"  Scanning {len(commits)} commits for bug fixes...")
    buggy_count = 0

    for commit in commits:
        if not is_bug_fix_commit(commit.message):
            continue
        if not commit.parents:
            continue

        parent = commit.parents[0]

        try:
            # Get files changed in this commit
            diffs = parent.diff(commit)
        except Exception:
            continue

        for diff in diffs:
            path = diff.a_path
            if not path.endswith(".py"):
                continue

            # Get the BUGGY version (before the fix = parent commit)
            content = get_file_content_at_commit(repo, parent, path)
            if not content or len(content) < 50:
                continue

            features = extract_features_from_content(content, path)
            if features:
                features["is_buggy"] = 1
                features["source"]   = repo_name
                features["path"]     = path
                samples.append(features)
                buggy_count += 1
                collected   += 1

    print(f"  ✓ Collected {buggy_count} buggy samples")

    # ── Collect CLEAN samples from stable files ───────────────────────────
    stable_files = get_stable_files(repo, commits)
    clean_count  = 0
    target_clean = buggy_count * 2  # 2:1 clean:buggy ratio

    for path in list(stable_files)[:target_clean * 2]:  # try more than needed
        content = get_file_content_at_commit(repo, repo.head.commit, path)
        if not content or len(content) < 50:
            continue

        features = extract_features_from_content(content, path)
        if features:
            features["is_buggy"] = 0
            features["source"]   = repo_name
            features["path"]     = path
            samples.append(features)
            clean_count += 1
            collected   += 1

        if clean_count >= target_clean:
            break

    print(f"  ✓ Collected {clean_count} clean samples")
    return collected


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    samples = []

    print(f"\n🔍 Mining {len(REPOS)} repositories...\n")

    for repo_url in tqdm(REPOS, desc="Repos"):
        print(f"\n📦 {repo_url}")
        count = mine_repo(repo_url, samples)
        print(f"  Total from this repo: {count}")

    if not samples:
        print("\n✗ No samples collected. Check your internet connection.")
        return

    df = pd.DataFrame(samples)

    # Drop metadata columns before saving features
    feature_cols = [c for c in df.columns if c not in ("is_buggy", "source", "path")]
    print(f"\n📊 Dataset Summary:")
    print(f"  Total samples:  {len(df)}")
    print(f"  Buggy samples:  {df['is_buggy'].sum()}")
    print(f"  Clean samples:  {(df['is_buggy'] == 0).sum()}")
    print(f"  Features:       {len(feature_cols)}")
    print(f"  Repos mined:    {df['source'].nunique()}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Dataset saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()