import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.recon_strategy.volume_root_connection import plan_root_connections
from lib.root_export import export_root_set


def test_root_export_preserves_connector_provenance():
    labels = np.zeros((9, 5, 5), np.int64); labels[4:] = 1
    solid = np.zeros_like(labels, bool); evidence = np.zeros_like(labels, np.uint8); conflict = np.zeros_like(labels, bool)
    low = np.zeros(3); high = np.array([.08, .04, .04])
    roots = np.array([[.03, .02, .02], [.05, .02, .02]], np.float32)
    plan = plan_root_connections(roots, labels, solid, evidence, conflict, low, high)
    result = export_root_set(plan, labels, solid, evidence, conflict, low, high)
    assert result["roots_world"].shape == roots.shape
    assert result["initial_connector"].sum() == 1
    assert result["growth_start_index"].tolist() == [1, 0]
