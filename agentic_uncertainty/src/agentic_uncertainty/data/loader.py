"""Load SWE-bench Pro dataset from HuggingFace."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import load_dataset

if TYPE_CHECKING:
    from .repo_context import RepoContext


@dataclass
class Task:
    """A single SWE-bench Pro task instance."""

    instance_id: str
    repo: str
    problem_statement: str
    hints_text: str
    base_commit: str
    patch: str  # Ground truth patch
    test_patch: str
    version: str
    fail_to_pass: str  # JSON string of tests that should go FAIL→PASS
    pass_to_pass: str  # JSON string of tests that should stay PASS
    repo_context: RepoContext | None = field(default=None, repr=False)

    @property
    def issue_description(self) -> str:
        """Get the full issue description for elicitation prompts.

        If repo_context is available, includes repository structure and
        relevant files as part of the description.
        """
        if self.repo_context is not None:
            context_str = self.repo_context.format_for_prompt()
            return f"{self.problem_statement}\n\n---\n\n{context_str}"
        return self.problem_statement

    @property
    def has_context(self) -> bool:
        """Check if repository context has been fetched."""
        return self.repo_context is not None

    async def with_context(self, max_depth: int = 3) -> Task:
        """Return a copy of this task with repository context fetched.

        Args:
            max_depth: Maximum depth for directory tree.

        Returns:
            New Task instance with repo_context populated.
        """
        if self.repo_context is not None:
            return self

        from .repo_context import get_repo_context

        context = await get_repo_context(
            repo=self.repo,
            commit=self.base_commit,
            problem_statement=self.problem_statement,
            max_depth=max_depth,
        )

        return Task(
            instance_id=self.instance_id,
            repo=self.repo,
            problem_statement=self.problem_statement,
            hints_text=self.hints_text,
            base_commit=self.base_commit,
            patch=self.patch,
            test_patch=self.test_patch,
            version=self.version,
            fail_to_pass=self.fail_to_pass,
            pass_to_pass=self.pass_to_pass,
            repo_context=context,
        )

    def with_context_sync(self, max_depth: int = 3) -> Task:
        """Synchronous version of with_context."""
        import asyncio

        return asyncio.run(self.with_context(max_depth))


class SWEBenchProLoader:
    """Load and iterate over SWE-bench Pro instances."""

    DATASET_NAME = "ScaleAI/SWE-bench_Pro"

    def __init__(self, split: str = "test"):
        """Initialize the loader.

        Args:
            split: Dataset split to use ("test" for the public set).
        """
        self.split = split
        self._dataset = None
        self._tasks: list[Task] | None = None

    def load(self) -> list[Task]:
        """Load the dataset and return list of Task instances."""
        if self._tasks is not None:
            return self._tasks

        if self._dataset is None:
            self._dataset = load_dataset(self.DATASET_NAME, split=self.split)

        tasks = []
        for item in self._dataset:
            task = Task(
                instance_id=item["instance_id"],
                repo=item["repo"],
                problem_statement=item["problem_statement"],
                hints_text=item.get("hints_text", ""),
                base_commit=item["base_commit"],
                patch=item["patch"],
                test_patch=item["test_patch"],
                version=item.get("version", ""),
                fail_to_pass=item.get("FAIL_TO_PASS", "[]"),
                pass_to_pass=item.get("PASS_TO_PASS", "[]"),
            )
            tasks.append(task)

        self._tasks = tasks
        return tasks

    def sample(
        self,
        n: int | None = None,
        seed: int = 42,
        repos: list[str] | None = None,
    ) -> list[Task]:
        """Sample a random subset of tasks.

        Args:
            n: Number of tasks to sample. If None, returns all tasks.
            seed: Random seed for reproducibility.
            repos: Optional list of repos to filter by (e.g., ["django/django"]).

        Returns:
            List of sampled Task instances.
        """
        tasks = self.load()

        # Filter by repos if specified
        if repos:
            tasks = [t for t in tasks if t.repo in repos]

        if n is None or n >= len(tasks):
            return tasks

        rng = random.Random(seed)
        return rng.sample(tasks, n)

    def __len__(self) -> int:
        """Return number of instances in the dataset."""
        return len(self.load())

    def __iter__(self):
        """Iterate over tasks."""
        return iter(self.load())


def load_ground_truth(eval_results_path: str | Path) -> dict[str, bool]:
    """Load ground truth from an eval_results.json file.

    Args:
        eval_results_path: Path to eval_results.json file from SWE-bench evaluation.
            Expected format: {"instance_id": true/false, ...}

    Returns:
        Dictionary mapping instance_id to resolved status (True/False).
    """
    with open(eval_results_path) as f:
        return json.load(f)
