from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx

from services.github import get_repo_tree, get_file_content
from services.language_router import analyze_file, detect_language
from services.predictor import predict_bug_probability
from services.explainer import explain_file

router = APIRouter(prefix="/analyze", tags=["analyze"])

class RepoRequest(BaseModel):
    repo_url: str
    max_files: int = 10
    explain: bool = False

def parse_github_url(url: str):
    parts = url.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError("Invalid GitHub URL")
    return parts[-2], parts[-1]

@router.post("/repo")
async def analyze_repo(request: RepoRequest):
    try:
        owner, repo = parse_github_url(request.repo_url)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL format")

    try:
        files = await get_repo_tree(owner, repo)
    except httpx.HTTPStatusError:
        raise HTTPException(status_code=404, detail="Repo not found or private")

    # Filter to only supported languages before capping max_files
    supported_files = [
        f for f in files if detect_language(f["path"]) is not None
    ]
    files_to_analyze = supported_files[:request.max_files]
    results = []

    for file in files_to_analyze:
        path     = file["path"]
        language = detect_language(path)

        if not language:
            continue

        try:
            content = await get_file_content(owner, repo, path)
        except Exception:
            continue

        analysis = analyze_file(content, path)
        if not analysis:
            continue

        bug_prediction = predict_bug_probability(
            analysis["metrics"], analysis["ast"], analysis["graph"]
        )

        explanation = None
        if request.explain:
            explanation = explain_file(
                path, analysis["metrics"], analysis["ast"],
                analysis["security"], analysis["graph"], bug_prediction
            )

        results.append({
            "path":           path,
            "language":       analysis["language"],
            "ast":            analysis["ast"],
            "metrics":        analysis["metrics"],
            "security":       analysis["security"],
            "graph":          analysis["graph"],
            "bug_prediction": bug_prediction,
            "explanation":    explanation,
        })

    # Sort by risk score descending
    results.sort(key=lambda x: x["metrics"]["risk_score"], reverse=True)

    return {
        "owner":          owner,
        "repo":           repo,
        "files_analyzed": len(results),
        "results":        results
    }   