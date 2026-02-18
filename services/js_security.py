import subprocess
import tempfile
import json
import os
import re

def run_js_security(content: str, path: str) -> list:
    """
    Run security checks on JS/TS files.
    Uses pattern matching for common vulnerabilities +
    semgrep if available.
    """
    issues = []

    # ── Pattern-based checks (always available) ───────────────────────────
    issues.extend(_pattern_checks(content))

    # ── Semgrep (if installed) ─────────────────────────────────────────────
    semgrep_issues = _run_semgrep(content, path)
    issues.extend(semgrep_issues)

    return issues


def _pattern_checks(content: str) -> list:
    """Fast regex-based security checks for common JS vulnerabilities"""
    issues = []
    lines  = content.split("\n")

    patterns = [
        # Injection risks
        (r"eval\s*\(", "HIGH",   "Use of eval() — potential code injection"),
        (r"innerHTML\s*=", "HIGH", "innerHTML assignment — potential XSS"),
        (r"document\.write\s*\(", "MEDIUM", "document.write() — potential XSS"),
        (r"dangerouslySetInnerHTML", "HIGH", "dangerouslySetInnerHTML — potential XSS"),

        # Hardcoded secrets
        (r"(?i)(password|secret|api_key|token)\s*=\s*['\"].+?['\"]",
         "HIGH", "Possible hardcoded secret or credential"),
        (r"(?i)Bearer\s+[A-Za-z0-9\-_]{20,}", "HIGH", "Possible hardcoded Bearer token"),

        # Insecure practices
        (r"Math\.random\(\)", "LOW", "Math.random() is not cryptographically secure"),
        (r"http://(?!localhost)", "LOW", "Non-HTTPS URL detected"),
        (r"console\.log\(", "LOW", "console.log() left in code — remove before production"),

        # Node.js specific
        (r"child_process", "MEDIUM", "child_process usage — validate all inputs"),
        (r"fs\.readFile|fs\.writeFile", "LOW", "File system access — ensure path is sanitized"),
        (r"\.exec\s*\(", "MEDIUM", "exec() usage — potential command injection"),
    ]

    for i, line in enumerate(lines):
        for pattern, severity, description in patterns:
            if re.search(pattern, line):
                issues.append({
                    "severity":    severity,
                    "confidence":  "MEDIUM",
                    "description": description,
                    "line":        i + 1,
                    "test_id":     "JS_PATTERN"
                })

    return issues


def _run_semgrep(content: str, path: str) -> list:
    """Run semgrep with javascript ruleset if available"""
    issues = []
    ext    = ".ts" if path.endswith(".ts") or path.endswith(".tsx") else ".js"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=ext, delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["semgrep", "--config", "p/javascript", "--json", tmp_path],
            capture_output=True, text=True, timeout=30,encoding="utf-8",errors="replace"
        )
        if result.stdout:
            data = json.loads(result.stdout)
            for finding in data.get("results", []):
                issues.append({
                    "severity":    finding.get("extra", {}).get("severity", "MEDIUM").upper(),
                    "confidence":  "HIGH",
                    "description": finding.get("extra", {}).get("message", ""),
                    "line":        finding.get("start", {}).get("line", 0),
                    "test_id":     finding.get("check_id", "SEMGREP")
                })
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass  # semgrep not installed or timed out — pattern checks still run
    finally:
        os.unlink(tmp_path)

    return issues