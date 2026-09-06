import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.recon_3d.build_spatial_cache_manifest import main


def test_spatial_cache_manifest_writes_physical_metadata(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle.npz"
    # Build the minimal contract expected by load_bundle through the public writer.
    from lib.recon_strategy.volume_partition import save_bundle, physical_sdf, EVIDENCE_CLASSES
    domain = np.ones((3, 3, 3), bool)
    spacing = np.array([.01, .02, .03])
    arrays = {"domain_mask": domain, "domain_sdf": physical_sdf(domain, spacing),
              "solid_mask": np.zeros_like(domain), "evidence_class": np.zeros_like(domain, np.uint8),
              "conflict": np.zeros_like(domain), "evidence_confidence": np.ones(domain.shape),
              "partition_labels": np.ones(domain.shape, np.int32), "partition_confidence": np.ones(domain.shape)}
    transform = np.eye(4)
    transform[:3, :3] = np.diag(1 / spacing)
    meta = {"version": 1, "axis_order": "XYZ", "units": "m", "origin": [0, 0, 0],
            "spacing": spacing.tolist(), "world_to_grid": transform.tolist(),
            "evidence_classes": EVIDENCE_CLASSES, "sources": ["test"], "depth_convention": "camera_z"}
    save_bundle(bundle, arrays, meta)
    output = tmp_path / "cache.json"
    monkeypatch.setattr(sys, "argv", ["build_spatial_cache_manifest", "--bundle", str(bundle), "--output", str(output)])
    main()
    report = json.loads(output.read_text())
    assert report["active_voxels"] == 27
    assert report["cache_fingerprint"]
