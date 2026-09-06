import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.ablation_manifest import build_ablation_manifest, compare_ablation_manifests


def test_ablation_manifest_requires_same_input_and_budget():
    a = build_ablation_manifest("input", seed=1, budget=10, configuration={"mode": "a"}, metrics={"loss": 1})
    b = build_ablation_manifest("input", seed=2, budget=10, configuration={"mode": "b"}, metrics={"loss": 2})
    assert compare_ablation_manifests([a, b])["comparable"]
    c = build_ablation_manifest("other", seed=1, budget=10, configuration={}, metrics={})
    assert not compare_ablation_manifests([a, c])["comparable"]


def test_ablation_rejects_zero_budget():
    with pytest.raises(ValueError):
        build_ablation_manifest("input", seed=1, budget=0, configuration={}, metrics={})
