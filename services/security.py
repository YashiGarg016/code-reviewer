import subprocess
import json
import tempfile
import os

def run_bandit(content: str, filename: str = "temp.py") -> list:
    """Run Bandit security scanner on a file's content"""
    issues = []

    # Write to a temp file since bandit needs a real file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["bandit", "-f", "json", "-q", tmp_path],
            capture_output=True, text=True,encoding="utf-8",      
            errors="replace"
        )
        if result.stdout:
            data = json.loads(result.stdout)
            for issue in data.get("results", []):
                issues.append({
                    "severity": issue["issue_severity"],       # LOW/MEDIUM/HIGH
                    "confidence": issue["issue_confidence"],
                    "description": issue["issue_text"],
                    "line": issue["line_number"],
                    "test_id": issue["test_id"],               # e.g. B105, B201
                })
    except Exception as e:
        issues = [{"error": str(e)}]
    finally:
        os.unlink(tmp_path)  # always clean up temp file

    return issues