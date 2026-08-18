from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_read_only_demo_has_all_critical_product_surfaces():
    html = (PROJECT_ROOT / "docs" / "index.html").read_text(encoding="utf-8")

    assert "A deterministic, auditable tabular AutoML and MLOps workflow" in html
    assert "Read-only demo" in html
    assert "Approval gate" in html
    assert 'data-view-panel="reports"' in html
    assert 'data-view-panel="artifacts"' in html
    assert "workflow_state.json" in html
    assert "114 untouched rows" in html
    assert "no data is uploaded" in html


def test_product_tour_is_local_and_runs_for_roughly_40_seconds():
    javascript = (PROJECT_ROOT / "docs" / "demo" / "assets" / "demo.js").read_text(
        encoding="utf-8"
    )
    html = (PROJECT_ROOT / "docs" / "index.html").read_text(encoding="utf-8")

    assert "delay: 39000" in javascript
    assert len(javascript.split("delay:")) - 1 == 8
    assert "http://" not in html
    assert "https://cdn" not in html
    assert "<iframe" not in html


def test_compact_architecture_diagram_documents_system_boundaries():
    diagram = (PROJECT_ROOT / "docs" / "product" / "architecture-overview.svg").read_text(
        encoding="utf-8"
    )

    assert "Agent-structured workflow" in diagram
    assert "Deterministic services" in diagram
    assert "Approval gate" in diagram
    assert "Run artifact store" in diagram
    assert "MLflow*" in diagram
