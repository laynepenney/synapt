"""User-configurable model selection and query defaults for the recall system.

Loads preferences from config files and environment variables.
Priority (highest wins): env vars → project config → global config → defaults.

Config locations:
  - Global: ~/.synapt/config.json
  - Project: .synapt/recall/config.json (relative to project root)

Example config:
  {
    "models": {
      "embedding": "all-MiniLM-L6-v2",
      "summarization": "google/flan-t5-base",
      "enrichment": "laynepro/t5-enrichment-v2",
      "consolidation": "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
      "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    },
    "backend": "auto",
    "max_tokens": 1500
  }
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default model names — these match the current hardcoded defaults
DEFAULTS = {
    "embedding": "all-MiniLM-L6-v2",
    "summarization": "google/flan-t5-base",
    "enrichment": "laynepro/t5-enrichment-v2",
    "consolidation": "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}

# Env var → config key mapping (model overrides)
_ENV_MAP = {
    "SYNAPT_SUMMARY_MODEL": "summarization",
    "SYNAPT_ENRICHMENT_MODEL": "enrichment",
    "SYNAPT_RERANKER_MODEL": "reranker",
    "SYNAPT_EMBEDDING_MODEL": "embedding",
}

# Default query parameters
DEFAULT_MAX_TOKENS = 1500
DEFAULT_SESSION_START_CONTINUITY = "automatic"
SESSION_START_CONTINUITY_MODES = {"off", "explicit", "automatic", "always"}

# Reverse lookup: config key → env var name
_KEY_TO_ENV = {v: k for k, v in _ENV_MAP.items()}


@dataclass
class RecallConfig:
    """Resolved model configuration."""

    models: dict[str, str] = field(default_factory=lambda: dict(DEFAULTS))
    backend: str = "auto"
    max_tokens: int = DEFAULT_MAX_TOKENS
    session_start_continuity: str = DEFAULT_SESSION_START_CONTINUITY

    def get_model(self, key: str) -> str:
        """Get a model name by key, with env var override."""
        # Check env var first (highest priority)
        env_var = _KEY_TO_ENV.get(key)
        if env_var:
            env_val = os.environ.get(env_var)
            if env_val:
                return env_val

        return self.models.get(key, DEFAULTS.get(key, ""))

    def get_max_tokens(self) -> int:
        """Get the configured max_tokens default, with env var override."""
        env_val = os.environ.get("SYNAPT_MAX_TOKENS")
        if env_val:
            try:
                return int(env_val)
            except ValueError:
                logger.warning("Invalid SYNAPT_MAX_TOKENS=%r, using %d", env_val, self.max_tokens)
        return self.max_tokens

    def get_session_start_continuity(self) -> str:
        """Return the SessionStart recovery policy, with env override."""
        value = os.environ.get(
            "SYNAPT_SESSION_START_CONTINUITY", self.session_start_continuity,
        ).strip().lower()
        if value not in SESSION_START_CONTINUITY_MODES:
            logger.warning(
                "Invalid SessionStart continuity mode %r, using %s",
                value,
                DEFAULT_SESSION_START_CONTINUITY,
            )
            return DEFAULT_SESSION_START_CONTINUITY
        return value

    def active_models(self) -> dict[str, str]:
        """Get all active model names (with env overrides applied)."""
        result = {}
        for key in DEFAULTS:
            result[key] = self.get_model(key)
        return result


# Module-level cache
_cached_config: RecallConfig | None = None
_cached_mtime: float = 0.0
_cached_project_path: str | None = None


def _find_project_config() -> str | None:
    """Find the project-level config file by walking up from cwd."""
    cwd = os.getcwd()
    for _ in range(20):  # Max depth
        candidate = os.path.join(cwd, ".synapt", "recall", "config.json")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(cwd)
        if parent == cwd:
            break
        cwd = parent
    return None


def _load_json(path: str) -> dict:
    """Load a JSON file, returning empty dict on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def load_config() -> RecallConfig:
    """Load and merge configuration from all sources.

    Priority: env vars → project config → global config → defaults.
    Results are cached and reloaded when config files change.
    """
    global _cached_config, _cached_mtime, _cached_project_path

    # Check if we need to reload
    global_path = os.path.expanduser("~/.synapt/config.json")

    # Reuse cached project path to avoid walking the directory tree every call
    if _cached_config is not None:
        project_path = _cached_project_path
    else:
        project_path = _find_project_config()
        _cached_project_path = project_path

    # Use max mtime of both config files for cache invalidation
    current_mtime = 0.0
    for path in (global_path, project_path):
        if path:
            try:
                current_mtime = max(current_mtime, os.path.getmtime(path))
            except OSError:
                pass

    if _cached_config is not None and current_mtime == _cached_mtime:
        return _cached_config

    # Start with defaults
    models = dict(DEFAULTS)

    # Layer 1: Global config
    global_data = _load_json(global_path)
    if "models" in global_data:
        models.update(global_data["models"])

    # Layer 2: Project config (overrides global)
    project_data = {}
    if project_path:
        project_data = _load_json(project_path)
        if "models" in project_data:
            models.update(project_data["models"])

    backend = (
        project_data.get("backend")
        or global_data.get("backend")
        or "auto"
    )

    # Env var for backend override
    env_backend = os.environ.get("SYNAPT_SUMMARY_BACKEND", "").lower()
    if env_backend:
        backend = env_backend

    # max_tokens: project overrides global overrides default
    max_tokens = DEFAULT_MAX_TOKENS
    if "max_tokens" in global_data:
        try:
            max_tokens = int(global_data["max_tokens"])
        except (ValueError, TypeError):
            logger.warning("Invalid max_tokens in global config, using default")
    if "max_tokens" in project_data:
        try:
            max_tokens = int(project_data["max_tokens"])
        except (ValueError, TypeError):
            logger.warning("Invalid max_tokens in project config, using default")

    continuity = DEFAULT_SESSION_START_CONTINUITY
    for data in (global_data, project_data):
        session_start = data.get("session_start", {})
        if isinstance(session_start, dict) and "continuity" in session_start:
            continuity = str(session_start["continuity"]).strip().lower()

    config = RecallConfig(
        models=models,
        backend=backend,
        max_tokens=max_tokens,
        session_start_continuity=continuity,
    )
    _cached_config = config
    _cached_mtime = current_mtime

    logger.debug("Loaded config: models=%s, backend=%s", models, backend)
    return config


def clear_config_cache() -> None:
    """Clear the config cache. Useful for testing."""
    global _cached_config, _cached_mtime, _cached_project_path
    _cached_config = None
    _cached_mtime = 0.0
    _cached_project_path = None
