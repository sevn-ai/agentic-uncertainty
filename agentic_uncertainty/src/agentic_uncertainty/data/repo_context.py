"""Fetch repository context from GitHub for pre-execution elicitation.

This module provides utilities to fetch repository structure and file contents
from GitHub, allowing models to see code context before predicting success.

Usage:
    from agentic_uncertainty.data.repo_context import get_repo_context

    context = await get_repo_context(
        repo="NodeBB/NodeBB",
        commit="abc123",
        problem_statement="The email validation is failing...",
    )
    print(context.tree)
    print(context.files)

Requires GITHUB_TOKEN environment variable for higher rate limits (5000/hour vs 60/hour).
"""

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

# GitHub API base URL
GITHUB_API = "https://api.github.com"

# Cache directory for repo context
CACHE_DIR = Path("data/repo_context_cache")

# Maximum depth for directory tree
DEFAULT_MAX_DEPTH = 3

# Maximum file size to fetch (100KB)
MAX_FILE_SIZE = 100_000

# Common file patterns to include for context
RELEVANT_PATTERNS = [
    r"README\.md$",
    r"package\.json$",
    r"pyproject\.toml$",
    r"setup\.py$",
    r"requirements\.txt$",
]


@dataclass
class RepoContext:
    """Repository context for pre-execution elicitation."""

    repo: str
    commit: str
    tree: str  # Directory structure as formatted string
    files: dict[str, str]  # path -> content for relevant files
    mentioned_files: list[str]  # Files mentioned in problem statement

    def format_for_prompt(self, max_tree_lines: int = 100, max_file_chars: int = 5000) -> str:
        """Format context for inclusion in a prompt.

        Args:
            max_tree_lines: Maximum lines to show in directory tree.
            max_file_chars: Maximum characters per file content.

        Returns:
            Formatted string suitable for prompt inclusion.
        """
        parts = []

        # Repository info
        parts.append(f"Repository: {self.repo}")
        parts.append(f"Commit: {self.commit[:12]}")
        parts.append("")

        # Directory structure
        tree_lines = self.tree.split("\n")
        if len(tree_lines) > max_tree_lines:
            tree_lines = tree_lines[:max_tree_lines] + [f"... ({len(tree_lines) - max_tree_lines} more entries)"]
        parts.append("## Directory Structure")
        parts.append("```")
        parts.append("\n".join(tree_lines))
        parts.append("```")
        parts.append("")

        # Relevant files
        if self.files:
            parts.append("## Relevant Files")
            for path, content in self.files.items():
                if len(content) > max_file_chars:
                    content = content[:max_file_chars] + f"\n... (truncated, {len(content)} chars total)"
                parts.append(f"\n### {path}")
                # Detect language for syntax highlighting
                lang = _get_language(path)
                parts.append(f"```{lang}")
                parts.append(content)
                parts.append("```")

        return "\n".join(parts)


def _get_language(path: str) -> str:
    """Get language identifier for syntax highlighting."""
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "jsx",
        ".tsx": "tsx",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".sh": "bash",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".rb": "ruby",
        ".php": "php",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".css": "css",
        ".html": "html",
        ".xml": "xml",
        ".sql": "sql",
        ".toml": "toml",
    }
    ext = Path(path).suffix.lower()
    return ext_map.get(ext, "")


def _get_github_headers() -> dict[str, str]:
    """Get headers for GitHub API requests."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _fetch_json(client: httpx.AsyncClient, url: str) -> dict[str, Any] | list[Any] | None:
    """Fetch JSON from GitHub API."""
    try:
        response = await client.get(url, headers=_get_github_headers())
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise


async def get_repo_tree(
    client: httpx.AsyncClient,
    repo: str,
    commit: str,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> str:
    """Fetch repository directory tree from GitHub.

    Args:
        client: HTTP client for requests.
        repo: Repository in "owner/name" format.
        commit: Commit SHA.
        max_depth: Maximum directory depth to traverse.

    Returns:
        Formatted directory tree string.
    """
    # Fetch tree recursively
    url = f"{GITHUB_API}/repos/{repo}/git/trees/{commit}?recursive=1"
    data = await _fetch_json(client, url)

    if not data or "tree" not in data:
        return "(Could not fetch repository tree)"

    # Build tree structure
    tree_items = data["tree"]
    lines = []

    for item in tree_items:
        path = item["path"]
        item_type = item["type"]  # "blob" or "tree"

        # Filter by depth
        depth = path.count("/")
        if depth >= max_depth:
            continue

        # Format with indentation
        indent = "  " * depth
        name = path.split("/")[-1]
        if item_type == "tree":
            lines.append(f"{indent}{name}/")
        else:
            lines.append(f"{indent}{name}")

    return "\n".join(sorted(lines))


async def get_file_content(
    client: httpx.AsyncClient,
    repo: str,
    commit: str,
    path: str,
) -> str | None:
    """Fetch a single file's content from GitHub.

    Args:
        client: HTTP client for requests.
        repo: Repository in "owner/name" format.
        commit: Commit SHA.
        path: File path within repository.

    Returns:
        File content as string, or None if not found.
    """
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}?ref={commit}"
    data = await _fetch_json(client, url)

    if not data:
        return None

    # Check file size
    size = data.get("size", 0)
    if size > MAX_FILE_SIZE:
        return f"(File too large: {size} bytes)"

    # Content is base64 encoded
    import base64

    content_b64 = data.get("content", "")
    if not content_b64:
        return None

    try:
        return base64.b64decode(content_b64).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return "(Binary file)"


def extract_mentioned_files(problem_statement: str) -> list[str]:
    """Extract file paths mentioned in a problem statement.

    Uses heuristics to find potential file paths:
    - Patterns like `path/to/file.py`
    - Code references in backticks
    - Common file extensions

    Args:
        problem_statement: The issue/task description.

    Returns:
        List of potential file paths.
    """
    mentioned = set()

    # Pattern for file paths (with extensions)
    file_pattern = r"[a-zA-Z0-9_\-./]+\.[a-zA-Z]{1,10}"

    # Find in backticks first (more reliable)
    backtick_pattern = r"`([^`]+)`"
    for match in re.finditer(backtick_pattern, problem_statement):
        content = match.group(1)
        if re.match(file_pattern, content) and "/" in content:
            mentioned.add(content)

    # Find general file references
    for match in re.finditer(file_pattern, problem_statement):
        path = match.group(0)
        # Filter out URLs and common false positives
        if (
            "/" in path
            and not path.startswith("http")
            and not path.startswith("www.")
            and len(path) < 100
        ):
            mentioned.add(path)

    return list(mentioned)


def _get_cache_path(repo: str, commit: str) -> Path:
    """Get cache file path for repo context."""
    safe_repo = repo.replace("/", "_")
    return CACHE_DIR / f"{safe_repo}_{commit[:12]}.json"


async def get_repo_context(
    repo: str,
    commit: str,
    problem_statement: str = "",
    max_depth: int = DEFAULT_MAX_DEPTH,
    include_mentioned: bool = True,
    use_cache: bool = True,
) -> RepoContext:
    """Fetch complete repository context for a task.

    Args:
        repo: Repository in "owner/name" format.
        commit: Commit SHA.
        problem_statement: Task description (used to find mentioned files).
        max_depth: Maximum depth for directory tree.
        include_mentioned: Whether to fetch files mentioned in problem statement.
        use_cache: Whether to use cached context if available.

    Returns:
        RepoContext with tree and relevant file contents.
    """
    import json

    # Check cache
    cache_path = _get_cache_path(repo, commit)
    if use_cache and cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
                return RepoContext(**data)
        except (json.JSONDecodeError, KeyError):
            pass

    # Fetch from GitHub
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get directory tree
        tree = await get_repo_tree(client, repo, commit, max_depth)

        # Extract and fetch mentioned files
        mentioned_files = extract_mentioned_files(problem_statement) if problem_statement else []
        files = {}

        if include_mentioned:
            for path in mentioned_files:
                content = await get_file_content(client, repo, commit, path)
                if content:
                    files[path] = content

        # Also fetch common config files for context
        for pattern in RELEVANT_PATTERNS:
            # Check if any file in tree matches
            for line in tree.split("\n"):
                name = line.strip().rstrip("/")
                if re.match(pattern, name):
                    content = await get_file_content(client, repo, commit, name)
                    if content and name not in files:
                        files[name] = content

    context = RepoContext(
        repo=repo,
        commit=commit,
        tree=tree,
        files=files,
        mentioned_files=mentioned_files,
    )

    # Cache result
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "repo": context.repo,
                    "commit": context.commit,
                    "tree": context.tree,
                    "files": context.files,
                    "mentioned_files": context.mentioned_files,
                },
                f,
                indent=2,
            )

    return context


# Synchronous wrapper for non-async contexts
def get_repo_context_sync(
    repo: str,
    commit: str,
    problem_statement: str = "",
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> RepoContext:
    """Synchronous version of get_repo_context."""
    import asyncio

    return asyncio.run(get_repo_context(repo, commit, problem_statement, max_depth))
