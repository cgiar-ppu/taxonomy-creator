"""Configuration loader and validator for Taxonomy Creator."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Default configuration values
DEFAULTS: Dict[str, Any] = {
    "model": {
        "provider": "anthropic",
        "name": "claude-sonnet-4-6",
        "max_tokens": 16384,
        "temperature": 0.1,
    },
    "input": {
        "file": "data/input.xlsx",
        "columns": {
            "title": "Title",
            "description": "Description",
        },
        "batch_size": 25,
    },
    "output": {
        "dir": "output",
        "formats": ["json", "graphml", "markdown", "html"],
    },
    "extraction": {
        "extract_concepts": True,
        "extract_entities": True,
        "extract_relationships": True,
        "extract_tags": True,
        "relationship_types": [
            "is_a",
            "part_of",
            "uses",
            "produces",
            "targets",
            "located_in",
            "collaborates_with",
            "addresses",
            "related_to",
        ],
    },
    "taxonomy": {
        "min_frequency": 2,
        "merge_threshold": 0.85,
        "max_depth": 4,
    },
}

# Approximate pricing per 1M tokens (input/output) as of 2025
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # Current Anthropic models (2026)
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},       # 1M context, 64K output
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},         # 1M context, 128K output
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},         # 1M context, 128K output
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},         # 200K context, 64K output
    # Legacy Anthropic models
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},  # deprecated, retiring June 2026
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # OpenAI models
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
}


class Config:
    """Application configuration loaded from YAML with defaults."""

    def __init__(self, config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> None:
        """Load configuration from YAML file, applying defaults for missing values.

        Args:
            config_path: Path to config.yaml. If None, searches standard locations.
            overrides: Dictionary of overrides to apply on top of loaded config.
        """
        self._data: Dict[str, Any] = self._deep_merge(DEFAULTS, {})

        if config_path is None:
            config_path = self._find_config()

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
            self._data = self._deep_merge(self._data, file_config)

        if overrides:
            self._data = self._deep_merge(self._data, overrides)

        self._resolve_paths()
        self._validate()

    def _find_config(self) -> Optional[str]:
        """Search for config.yaml in standard locations."""
        candidates = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _resolve_paths(self) -> None:
        """Resolve relative paths to absolute paths based on project root."""
        project_root = Path(__file__).parent.parent
        input_file = self._data["input"]["file"]
        if not Path(input_file).is_absolute():
            self._data["input"]["file"] = str(project_root / input_file)

        output_dir = self._data["output"]["dir"]
        if not Path(output_dir).is_absolute():
            self._data["output"]["dir"] = str(project_root / output_dir)

    def _validate(self) -> None:
        """Validate configuration values."""
        provider = self._data["model"]["provider"]
        if provider not in ("anthropic", "openai"):
            raise ValueError(f"Unsupported model provider: {provider}")

        batch_size = self._data["input"]["batch_size"]
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got: {batch_size}")

        max_depth = self._data["taxonomy"]["max_depth"]
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError(f"max_depth must be a positive integer, got: {max_depth}")

        threshold = self._data["taxonomy"]["merge_threshold"]
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"merge_threshold must be between 0 and 1, got: {threshold}")

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries. Override values take precedence."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    # --- Accessors ---

    @property
    def model_provider(self) -> str:
        return self._data["model"]["provider"]

    @property
    def model_name(self) -> str:
        return self._data["model"]["name"]

    @property
    def max_tokens(self) -> int:
        return self._data["model"]["max_tokens"]

    @property
    def temperature(self) -> float:
        return self._data["model"]["temperature"]

    @property
    def input_file(self) -> str:
        return self._data["input"]["file"]

    @property
    def title_column(self) -> str:
        return self._data["input"]["columns"]["title"]

    @property
    def description_column(self) -> str:
        return self._data["input"]["columns"]["description"]

    @property
    def batch_size(self) -> int:
        return self._data["input"]["batch_size"]

    @property
    def output_dir(self) -> str:
        return self._data["output"]["dir"]

    @property
    def output_formats(self) -> List[str]:
        return self._data["output"]["formats"]

    @property
    def extraction(self) -> Dict[str, Any]:
        return self._data["extraction"]

    @property
    def relationship_types(self) -> List[str]:
        return self._data["extraction"]["relationship_types"]

    @property
    def taxonomy(self) -> Dict[str, Any]:
        return self._data["taxonomy"]

    @property
    def min_frequency(self) -> int:
        return self._data["taxonomy"]["min_frequency"]

    @property
    def merge_threshold(self) -> float:
        return self._data["taxonomy"]["merge_threshold"]

    @property
    def max_depth(self) -> int:
        return self._data["taxonomy"]["max_depth"]

    def get_model_pricing(self) -> Dict[str, float]:
        """Get pricing for the configured model. Returns input/output cost per 1M tokens."""
        return MODEL_PRICING.get(
            self.model_name,
            {"input": 3.0, "output": 15.0},  # conservative default
        )

    def get_api_key(self) -> str:
        """Get the API key from environment variables."""
        if self.model_provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY", "")
        else:
            key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise EnvironmentError(
                f"API key not set. Please set {'ANTHROPIC_API_KEY' if self.model_provider == 'anthropic' else 'OPENAI_API_KEY'} "
                f"in your environment or .env file."
            )
        return key

    def __repr__(self) -> str:
        return f"Config(provider={self.model_provider}, model={self.model_name}, batch_size={self.batch_size})"
