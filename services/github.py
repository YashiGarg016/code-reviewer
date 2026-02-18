import httpx
import os
import base64
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

# Add a generous timeout — 30s connect, 60s read
TIMEOUT = httpx.Timeout(connect=30.0, read=60.0, write=10.0, pool=10.0)

SUPPORTED_EXTENSIONS = (".py", ".js", ".ts", ".jsx", ".tsx")

async def get_repo_tree(owner: str, repo: str) -> list:
    """Fetch all supported files in a repo recursively"""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()

    files = [
        item for item in data.get("tree", [])
        if item["type"] == "blob"
        and item["path"].endswith(SUPPORTED_EXTENSIONS)
    ]
    return files

async def get_file_content(owner: str, repo: str, path: str) -> str:
    """Fetch raw content of a single file"""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()

    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return content