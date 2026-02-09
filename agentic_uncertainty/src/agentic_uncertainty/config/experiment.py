"""Experiment configuration loaded from YAML files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str  # Model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4o", "gemini-2.5-flash", "grok-3")
    provider: Literal["anthropic", "openai", "gemini", "grok"]
    api_key_env: str | None = None  # Env var name for API key (auto-detected if None)
    base_url_env: str | None = None  # Env var name for base URL (optional)
    temperature: float = 0.0
    sampling_temperature: float = 0.9
    # Gemini-specific configuration
    project_env: str | None = None  # Env var name for GCP project (default: GOOGLE_CLOUD_PROJECT)
    location: str = "us-central1"  # Google Cloud location


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    name: str  # Experiment name (used for output directory)
    models: list[ModelConfig]
    methods: list[str] = field(
        default_factory=lambda: ["direct", "calibrated", "blockers"]
    )
    num_samples: int | None = None  # None means use all instances
    seed: int = 42
    repos: list[str] | None = None
    instance_ids_file: str | None = None  # Path to JSON file with instance IDs


@dataclass
class MultiModelExperimentConfig:
    """Top-level configuration for multi-model experiments."""

    experiments: list[ExperimentConfig]
    output_base_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: Path | str) -> "MultiModelExperimentConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Parsed MultiModelExperimentConfig.
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        experiments = []
        for exp_data in data.get("experiments", []):
            models = []
            for model_data in exp_data.get("models", []):
                models.append(
                    ModelConfig(
                        name=model_data["name"],
                        provider=model_data["provider"],
                        api_key_env=model_data.get("api_key_env"),
                        base_url_env=model_data.get("base_url_env"),
                        temperature=model_data.get("temperature", 0.0),
                        sampling_temperature=model_data.get("sampling_temperature", 0.9),
                        project_env=model_data.get("project_env"),
                        location=model_data.get("location", "us-central1"),
                    )
                )

            experiments.append(
                ExperimentConfig(
                    name=exp_data["name"],
                    models=models,
                    methods=exp_data.get(
                        "methods", ["direct", "calibrated", "blockers"]
                    ),
                    num_samples=exp_data.get("num_samples"),
                    seed=exp_data.get("seed", 42),
                    repos=exp_data.get("repos"),
                    instance_ids_file=exp_data.get("instance_ids_file"),
                )
            )

        return cls(
            experiments=experiments,
            output_base_dir=data.get("output_base_dir", "results"),
        )

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the YAML configuration file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "output_base_dir": self.output_base_dir,
            "experiments": [
                {
                    "name": exp.name,
                    "models": [
                        {
                            "name": model.name,
                            "provider": model.provider,
                            **({"api_key_env": model.api_key_env} if model.api_key_env else {}),
                            **({"base_url_env": model.base_url_env} if model.base_url_env else {}),
                            "temperature": model.temperature,
                            "sampling_temperature": model.sampling_temperature,
                            **({"project_env": model.project_env} if model.project_env else {}),
                            **({"location": model.location} if model.location != "us-central1" else {}),
                        }
                        for model in exp.models
                    ],
                    "methods": exp.methods,
                    **({"num_samples": exp.num_samples} if exp.num_samples else {}),
                    "seed": exp.seed,
                    **({"repos": exp.repos} if exp.repos else {}),
                    **({"instance_ids_file": exp.instance_ids_file} if exp.instance_ids_file else {}),
                }
                for exp in self.experiments
            ],
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
