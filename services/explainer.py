from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(path: str, metrics: dict, ast_info: dict,
                 security: list, graph: dict, bug_prediction: dict) -> str:

    complex_fns = metrics.get("complex_functions", [])
    complex_fn_text = "\n".join(
        f"  - {fn['name']} (complexity: {fn['complexity']}, line {fn['line']})"
        for fn in complex_fns
    ) or "  None"

    security_text = "\n".join(
        f"  - [{issue.get('severity', 'UNKNOWN')}] {issue.get('description', '')} (line {issue.get('line', '?')})"
        for issue in security[:5]
    ) or "  None"

    return f"""You are an expert software engineer performing a code review.
Analyze this file based on static analysis results and provide actionable feedback.

FILE: {path}

METRICS:
- Risk Score: {metrics.get('risk_score', 0)}/100
- Cyclomatic Complexity (avg): {metrics.get('avg_complexity', 0)}
- Cyclomatic Complexity (max): {metrics.get('max_complexity', 0)}
- Maintainability Index: {metrics.get('maintainability_index', 0)}/100
- Total Lines: {metrics.get('total_lines', 0)}

COMPLEX FUNCTIONS:
{complex_fn_text}

SECURITY ISSUES:
{security_text}

CALL GRAPH:
- Nodes: {graph.get('num_nodes', 0)}, Edges: {graph.get('num_edges', 0)}
- Most connected function: {graph.get('most_connected_node', 'none')}
- Deep nesting count: {graph.get('deep_nesting_count', 0)}
- Clustering coefficient: {graph.get('clustering_coefficient', 0)}

ML BUG PREDICTION:
- Bug probability: {bug_prediction.get('bug_probability', 0) * 100:.1f}%
- Confidence: {bug_prediction.get('confidence', 'unknown')}

CODE STRUCTURE:
- Functions: {len(ast_info.get('functions', []))}
- Classes: {len(ast_info.get('classes', []))}
- Imports: {len(ast_info.get('imports', []))}

Based on this analysis, provide a code review with exactly this structure:

## Summary
2-3 sentences on the overall health of this file.

## Key Issues
List the 3 most important problems, each with a concrete fix suggestion.

## Quick Wins
2 small improvements that would immediately improve code quality.

## Verdict
One line: GREEN (healthy), YELLOW (needs attention), or RED (critical issues).

Be specific, actionable, and concise. Do not hallucinate code you haven't seen.
Acknowledge you're working from metrics, not the actual source code."""


def explain_file(path: str, metrics: dict, ast_info: dict,
                 security: list, graph: dict, bug_prediction: dict) -> dict:
    try:
        prompt = build_prompt(path, metrics, ast_info, security, graph, bug_prediction)

        message = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        explanation = message.choices[0].message.content

        # Parse verdict
        verdict = "YELLOW"
        if "## Verdict" in explanation:
            verdict_section = explanation.split("## Verdict")[-1].strip()
            if "GREEN" in verdict_section.upper():
                verdict = "GREEN"
            elif "RED" in verdict_section.upper():
                verdict = "RED"

        return {
            "explanation": explanation,
            "verdict": verdict,
            "tokens_used": message.usage.total_tokens
        }

    except Exception as e:
        return {
            "explanation": f"Could not generate explanation: {str(e)}",
            "verdict": "UNKNOWN",
            "tokens_used": 0
        }