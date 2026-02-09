"""Batch instance loading and configuration utilities."""

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minisweagent.utils.files import load_file
from minisweagent.utils.log import logger


@dataclass
class BatchInstance:
    """A single instance in a batch of instances."""

    instance_id: str = field(metadata={"help": "Unique identifier for the instance."})
    problem_statement: str = field(
        metadata={"help": "Description of the problem to solve."}
    )
    base_commit: str = field(
        metadata={"help": "Git commit hash to use as the starting point."}
    )
    image_name: str = field(
        default="", metadata={"help": "Docker image name for the environment."}
    )
    repo_name: str = field(
        default="app",
        metadata={
            "help": "Name of the directory under `/` where the repository is located."
        },
    )
    extra_fields: dict[str, Any] = field(default_factory=dict)


def _slice_spec_to_slice(slice_spec: str) -> slice:
    """Convert a string slice specification to a slice object."""
    if slice_spec == "":
        return slice(None)
    parts = slice_spec.split(":")
    values = [None if p == "" else int(p) for p in parts]
    if len(parts) == 1:
        return slice(values[0])
    if len(parts) == 2:
        return slice(values[0], values[1])
    if len(parts) == 3:
        return slice(values[0], values[1], values[2])
    msg = (
        f"Invalid slice specification: {slice_spec!r}. "
        "Expected format: stop or start:stop or start:stop:step"
    )
    raise ValueError(msg)


def filter_instances(
    instances: list[BatchInstance],
    *,
    filter_spec: str = ".*",
    slice_spec: str = "",
    shuffle: bool = False,
) -> list[BatchInstance]:
    """Filter, slice, and optionally shuffle instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x.instance_id)
        random.seed(42)
        random.shuffle(instances)

    before_filter = len(instances)
    instances = [
        instance
        for instance in instances
        if re.match(filter_spec, instance.instance_id)
    ]
    after_filter = len(instances)
    if before_filter != after_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")

    if slice_spec:
        instances = instances[_slice_spec_to_slice(slice_spec)]
        after_slice = len(instances)
        if before_filter != after_slice:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")

    return instances


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the Docker image name for a SWE-bench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = (
            f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        )
    return image_name


def load_instances_from_file(
    path: Path,
    *,
    filter_spec: str = ".*",
    slice_spec: str = "",
    shuffle: bool = False,
) -> list[BatchInstance]:
    """Load instances from a file (JSON, JSONL, or YAML)."""
    raw_instances = load_file(path)
    instances = []
    for inst in raw_instances:
        instances.append(
            BatchInstance(
                instance_id=inst["instance_id"],
                problem_statement=inst["problem_statement"],
                image_name=inst["image_name"],
                repo_name=inst["repo_name"],
                base_commit=inst["base_commit"],
                extra_fields={
                    k: v
                    for k, v in inst.items()
                    if k
                    not in [
                        "instance_id",
                        "problem_statement",
                        "image_name",
                        "repo_name",
                        "base_commit",
                    ]
                },
            )
        )
    return filter_instances(
        instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle
    )


def load_instances_from_huggingface(
    dataset_name: str,
    split: str = "dev",
    *,
    filter_spec: str = ".*",
    slice_spec: str = "",
    shuffle: bool = False,
) -> list[BatchInstance]:
    """Load instances from a HuggingFace dataset."""
    from datasets import load_dataset

    ds = list(load_dataset(dataset_name, split=split))
    instances = []
    for inst in ds:
        instances.append(
            BatchInstance(
                instance_id=inst["instance_id"],
                problem_statement=inst["problem_statement"],
                image_name=inst["image_name"],
                repo_name=inst["repo_name"],
                base_commit=inst["base_commit"],
                extra_fields={
                    k: v
                    for k, v in inst.items()
                    if k
                    not in [
                        "instance_id",
                        "problem_statement",
                        "image_name",
                        "repo_name",
                        "base_commit",
                    ]
                },
            )
        )
    return filter_instances(
        instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle
    )


def load_swebench_instances(
    subset: str = "lite",
    split: str = "dev",
    *,
    filter_spec: str = ".*",
    slice_spec: str = "",
    shuffle: bool = False,
) -> list[BatchInstance]:
    """Load instances from SWE-bench."""
    from datasets import load_dataset

    dataset_mapping = {
        "full": "princeton-nlp/SWE-Bench",
        "verified": "princeton-nlp/SWE-Bench_Verified",
        "lite": "princeton-nlp/SWE-Bench_Lite",
        "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
        "multilingual": "swe-bench/SWE-Bench_Multilingual",
        "smith": "SWE-bench/SWE-smith",
        "_test": "klieret/swe-bench-dummy-test-dataset",
    }

    dataset_path = dataset_mapping.get(subset, subset)
    ds = list(load_dataset(dataset_path, split=split))

    instances = []
    for inst in ds:
        image_name = get_swebench_docker_image_name(inst)
        instances.append(
            BatchInstance(
                instance_id=inst["instance_id"],
                problem_statement=inst["problem_statement"],
                image_name=image_name,
                repo_name="testbed",
                base_commit=inst.get("base_commit", "HEAD"),
                extra_fields={
                    k: v
                    for k, v in inst.items()
                    if k
                    not in [
                        "instance_id",
                        "problem_statement",
                        "image_name",
                        "repo_name",
                        "base_commit",
                    ]
                },
            )
        )
    return filter_instances(
        instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle
    )
