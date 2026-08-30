"""Tests for user-configurable model selection."""

from __future__ import annotations

import json

import pytest

from synapt.recall.config import (
    DEFAULTS,
    RecallConfig,
    clear_config_cache,
    load_config,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear config cache before each test."""
    clear_config_cache()
    yield
    clear_config_cache()


class TestRecallConfig:
    """Test the RecallConfig dataclass."""

    def test_defaults(self):
        cfg = RecallConfig()
        assert cfg.get_model("embedding") == DEFAULTS["embedding"]
        assert cfg.get_model("summarization") == DEFAULTS["summarization"]
        assert cfg.get_model("enrichment") == DEFAULTS["enrichment"]
        assert cfg.backend == "auto"
        assert cfg.get_session_start_continuity() == "automatic"

    def test_session_start_continuity_env_override(self, monkeypatch):
        monkeypatch.setenv("SYNAPT_SESSION_START_CONTINUITY", "explicit")
        assert RecallConfig().get_session_start_continuity() == "explicit"

    def test_invalid_session_start_continuity_falls_back(self):
        cfg = RecallConfig(session_start_continuity="surprise-me")
        assert cfg.get_session_start_continuity() == "automatic"

    def test_custom_model(self):
        cfg = RecallConfig(models={**DEFAULTS, "embedding": "custom-model"})
        assert cfg.get_model("embedding") == "custom-model"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("SYNAPT_SUMMARY_MODEL", "google/flan-t5-large")
        cfg = RecallConfig()
        assert cfg.get_model("summarization") == "google/flan-t5-large"

    def test_env_override_enrichment(self, monkeypatch):
        monkeypatch.setenv("SYNAPT_ENRICHMENT_MODEL", "custom/enrichment")
        cfg = RecallConfig()
        assert cfg.get_model("enrichment") == "custom/enrichment"

    def test_env_override_embedding(self, monkeypatch):
        monkeypatch.setenv("SYNAPT_EMBEDDING_MODEL", "all-MiniLM-L12-v2")
        cfg = RecallConfig()
        assert cfg.get_model("embedding") == "all-MiniLM-L12-v2"

    def test_env_override_reranker(self, monkeypatch):
        monkeypatch.setenv("SYNAPT_RERANKER_MODEL", "custom/reranker")
        cfg = RecallConfig()
        assert cfg.get_model("reranker") == "custom/reranker"

    def test_active_models_includes_all(self):
        cfg = RecallConfig()
        models = cfg.active_models()
        for key in DEFAULTS:
            assert key in models

    def test_query_freshness_env_overrides_configured_values(self, monkeypatch):
        cfg = RecallConfig(
            query_freshness={
                "age_threshold_seconds": 120.0,
                "byte_trigger": 2048.0,
                "step_bytes": 4096.0,
                "byte_cap": 8192.0,
                "wall_seconds": 2.0,
            }
        )
        monkeypatch.setenv("SYNAPT_QUERY_FRESHNESS_BYTE_CAP", "16384")
        monkeypatch.setenv("SYNAPT_QUERY_FRESHNESS_WALL_SECONDS", "3.5")

        values = cfg.get_query_freshness()

        assert values["age_threshold_seconds"] == 120.0
        assert values["byte_cap"] == 16384.0
        assert values["wall_seconds"] == 3.5

    def test_unknown_key_returns_empty(self):
        cfg = RecallConfig()
        assert cfg.get_model("nonexistent") == ""


class TestLoadConfig:
    """Test config loading from files."""

    def test_loads_defaults_when_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.get_model("embedding") == DEFAULTS["embedding"]

    def test_loads_global_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)

        config_dir = tmp_path / ".synapt"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({
            "models": {"embedding": "custom-global-model"}
        }))

        cfg = load_config()
        assert cfg.get_model("embedding") == "custom-global-model"
        # Other models stay at defaults
        assert cfg.get_model("summarization") == DEFAULTS["summarization"]

    def test_project_config_overrides_global(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)

        # Global config
        global_dir = tmp_path / ".synapt"
        global_dir.mkdir()
        (global_dir / "config.json").write_text(json.dumps({
            "models": {"embedding": "global-model", "summarization": "global-summary"}
        }))

        # Project config
        project_dir = tmp_path / ".synapt" / "recall"
        project_dir.mkdir(parents=True)
        (project_dir / "config.json").write_text(json.dumps({
            "models": {"embedding": "project-model"}
        }))

        cfg = load_config()
        assert cfg.get_model("embedding") == "project-model"
        assert cfg.get_model("summarization") == "global-summary"

    def test_project_config_controls_session_start_continuity(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        project_dir = tmp_path / ".synapt" / "recall"
        project_dir.mkdir(parents=True)
        (project_dir / "config.json").write_text(json.dumps({
            "session_start": {"continuity": "explicit"}
        }))

        assert load_config().get_session_start_continuity() == "explicit"

    def test_project_config_controls_query_freshness(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        project_dir = tmp_path / ".synapt" / "recall"
        project_dir.mkdir(parents=True)
        (project_dir / "config.json").write_text(
            json.dumps(
                {
                    "query_freshness": {
                        "age_threshold_seconds": 45,
                        "byte_trigger": 1024,
                    }
                }
            )
        )

        values = load_config().get_query_freshness()

        assert values["age_threshold_seconds"] == 45.0
        assert values["byte_trigger"] == 1024.0
        assert values["step_bytes"] == 4 * 1024 * 1024

    def test_env_var_overrides_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SYNAPT_SUMMARY_MODEL", "env-model")

        # Config file
        config_dir = tmp_path / ".synapt"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({
            "models": {"summarization": "file-model"}
        }))

        cfg = load_config()
        # Env var wins
        assert cfg.get_model("summarization") == "env-model"

    def test_backend_from_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)

        config_dir = tmp_path / ".synapt"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({
            "backend": "transformers"
        }))

        cfg = load_config()
        assert cfg.backend == "transformers"

    def test_backend_env_overrides_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SYNAPT_SUMMARY_BACKEND", "mlx")

        config_dir = tmp_path / ".synapt"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({
            "backend": "transformers"
        }))

        cfg = load_config()
        assert cfg.backend == "mlx"

    def test_config_caching(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)

        cfg1 = load_config()
        cfg2 = load_config()
        assert cfg1 is cfg2

    def test_malformed_json_uses_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)

        config_dir = tmp_path / ".synapt"
        config_dir.mkdir()
        (config_dir / "config.json").write_text("not valid json{{{")

        cfg = load_config()
        assert cfg.get_model("embedding") == DEFAULTS["embedding"]


class TestRouterConfigIntegration:
    """Test that the model router uses config."""

    def test_get_encoder_decoder_model_uses_config(self, tmp_path, monkeypatch):
        from synapt.recall._model_router import get_encoder_decoder_model

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        monkeypatch.chdir(tmp_path)

        config_dir = tmp_path / ".synapt"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({
            "models": {"summarization": "google/flan-t5-large"}
        }))

        assert get_encoder_decoder_model() == "google/flan-t5-large"
