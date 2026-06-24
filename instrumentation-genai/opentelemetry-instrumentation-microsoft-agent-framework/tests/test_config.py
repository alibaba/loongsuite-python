"""Tests for config env-var parsing."""

from __future__ import annotations

import importlib

from opentelemetry.instrumentation.microsoft_agent_framework import config


def _reload():
    importlib.reload(config)
    return config


def test_defaults(monkeypatch):
    monkeypatch.delenv("ARMS_MAF_INSTRUMENTATION_ENABLED", raising=False)
    monkeypatch.delenv("ARMS_MAF_SENSITIVE_DATA_ENABLED", raising=False)
    monkeypatch.delenv("ARMS_MAF_REACT_STEP_ENABLED", raising=False)
    monkeypatch.delenv("ARMS_MAF_SLOW_THRESHOLD_MS", raising=False)
    monkeypatch.delenv("ARMS_MAF_METRICS_ENABLED", raising=False)
    cfg = _reload()
    assert cfg.is_instrumentation_enabled() is True
    assert cfg.is_sensitive_data_enabled() is False
    assert cfg.is_react_step_enabled() is False
    assert cfg.is_metrics_enabled() is True
    assert cfg.get_slow_threshold_ms() == 1000


def test_explicit_values(monkeypatch):
    monkeypatch.setenv("ARMS_MAF_SENSITIVE_DATA_ENABLED", "true")
    monkeypatch.setenv("ARMS_MAF_REACT_STEP_ENABLED", "1")
    monkeypatch.setenv("ARMS_MAF_SLOW_THRESHOLD_MS", "2500")
    monkeypatch.setenv("ARMS_MAF_METRICS_ENABLED", "off")
    cfg = _reload()
    assert cfg.is_sensitive_data_enabled() is True
    assert cfg.is_react_step_enabled() is True
    assert cfg.get_slow_threshold_ms() == 2500
    assert cfg.is_metrics_enabled() is False
