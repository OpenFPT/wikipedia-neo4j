"""Tests for ablation studies."""

from __future__ import annotations

import pytest

from scripts.run_ablation import (
    ABLATION_CONFIGS,
    AblationConfig,
    AblationResult,
    print_ablation_report,
)
from src.evaluation import EvalMetrics


def test_ablation_config_dataclass():
    """Test AblationConfig dataclass."""
    config = AblationConfig(
        name="test",
        description="Test config",
        multi_hop_expansion=True,
        use_reranking=True,
        use_graph=True,
    )
    assert config.name == "test"
    assert config.description == "Test config"
    assert config.multi_hop_expansion is True
    assert config.use_reranking is True
    assert config.use_graph is True


def test_ablation_result_dataclass():
    """Test AblationResult dataclass."""
    config = AblationConfig(
        name="test",
        description="Test",
        multi_hop_expansion=True,
        use_reranking=True,
        use_graph=True,
    )
    metrics = EvalMetrics(
        total=10,
        context_hit_rate=0.8,
        mrr=0.75,
        avg_latency_ms=100.0,
    )
    result = AblationResult(config=config, metrics=metrics)
    assert result.config.name == "test"
    assert result.metrics.context_hit_rate == 0.8


def test_ablation_configs_defined():
    """Test that ablation configurations are properly defined."""
    assert len(ABLATION_CONFIGS) == 5
    assert ABLATION_CONFIGS[0].name == "full_hybrid"
    assert ABLATION_CONFIGS[1].name == "no_reranking"
    assert ABLATION_CONFIGS[2].name == "no_multihop"
    assert ABLATION_CONFIGS[3].name == "graph_only"
    assert ABLATION_CONFIGS[4].name == "text_only"


def test_ablation_configs_properties():
    """Test ablation configuration properties."""
    full_hybrid = ABLATION_CONFIGS[0]
    assert full_hybrid.multi_hop_expansion is True
    assert full_hybrid.use_reranking is True
    assert full_hybrid.use_graph is True

    text_only = ABLATION_CONFIGS[4]
    assert text_only.multi_hop_expansion is False
    assert text_only.use_reranking is False
    assert text_only.use_graph is False


def test_print_ablation_report_empty():
    """Test ablation report with empty results."""
    report = print_ablation_report([])
    assert "Ablation Study Results" in report


def test_print_ablation_report_single():
    """Test ablation report with single result."""
    config = AblationConfig(
        name="test",
        description="Test config",
        multi_hop_expansion=True,
        use_reranking=True,
        use_graph=True,
    )
    metrics = EvalMetrics(
        total=10,
        context_hit_rate=0.8,
        mrr=0.75,
        avg_latency_ms=100.0,
    )
    result = AblationResult(config=config, metrics=metrics)

    report = print_ablation_report([result])
    assert "Ablation Study Results" in report
    assert "test" in report
    assert "0.800" in report
    assert "0.750" in report


def test_print_ablation_report_multiple():
    """Test ablation report with multiple results."""
    configs = [
        AblationConfig(
            name="full",
            description="Full system",
            multi_hop_expansion=True,
            use_reranking=True,
            use_graph=True,
        ),
        AblationConfig(
            name="no_rerank",
            description="Without reranking",
            multi_hop_expansion=True,
            use_reranking=False,
            use_graph=True,
        ),
    ]
    metrics_list = [
        EvalMetrics(total=10, context_hit_rate=0.8, mrr=0.75, avg_latency_ms=100.0),
        EvalMetrics(total=10, context_hit_rate=0.7, mrr=0.65, avg_latency_ms=80.0),
    ]
    results = [AblationResult(config=c, metrics=m) for c, m in zip(configs, metrics_list)]

    report = print_ablation_report(results)
    assert "Ablation Study Results" in report
    assert "Component Contributions" in report
    assert "full" in report
    assert "no_rerank" in report
    assert "-10.0%" in report  # Hit rate delta
