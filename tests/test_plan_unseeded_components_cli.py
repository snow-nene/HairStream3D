import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vis.plan_unseeded_components import main


def test_cli_rejects_sparse_seed_only_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle.npz"
    domain = np.ones((2, 2, 2), bool)
    np.savez(bundle, domain_mask=domain, partition_labels=np.zeros_like(domain), seeds=np.ones_like(domain))
    with pytest.raises(ValueError, match="authoritative"):
        monkeypatch.setattr(sys, "argv", ["plan", "--bundle", str(bundle), "--output", str(tmp_path / "out.json")])
        main()
