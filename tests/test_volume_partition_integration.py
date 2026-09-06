"""体积契约、深度种子及整线段门禁的行为回归。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.recon_strategy.volume_partition import (
    EVIDENCE_CLASSES, physical_sdf, validate_bundle, save_bundle, load_bundle,
    surface_partition_seeds, propagate_partitions,
)
from lib.recon_strategy.volume_segment_guard import (
    first_invalid_segment_fraction, constrain_volume_segments, audit_volume_strands,
)


class VolumeContractTests(unittest.TestCase):
    def test_contract_roundtrip_and_rejects_ambiguous_coordinates(self):
        domain = np.ones((3, 4, 5), bool)
        spacing = np.array([0.01, 0.02, 0.03])
        transform = np.eye(4)
        transform[:3, :3] = np.diag(1 / spacing)
        meta = dict(version=1, axis_order="XYZ", units="m", origin=[0, 0, 0],
                    spacing=spacing.tolist(), world_to_grid=transform.tolist(),
                    evidence_classes=EVIDENCE_CLASSES, sources=["synthetic"],
                    depth_convention="camera_z")
        arrays = dict(domain_mask=domain, domain_sdf=physical_sdf(domain, spacing),
                      solid_mask=~domain, evidence_class=np.zeros(domain.shape, np.uint8),
                      conflict=~domain, evidence_confidence=np.ones(domain.shape),
                      partition_labels=np.ones(domain.shape, np.int32),
                      partition_confidence=np.ones(domain.shape))
        output = Path(__file__).parent / "outputs/volume_partition_integration"
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            path = Path(tmp) / "bundle.npz"
            save_bundle(path, arrays, meta)
            restored, metadata = load_bundle(path)
            np.testing.assert_array_equal(restored["domain_mask"], domain)
            self.assertEqual(metadata, meta)
        broken = dict(meta, axis_order="ZYX")
        with self.assertRaises(ValueError):
            validate_bundle(arrays, broken)
        arrays["solid_mask"][1, 1, 1] = True
        with self.assertRaisesRegex(ValueError, "intersects solid"):
            validate_bundle(arrays, meta)

    def test_surface_band_does_not_extrude_to_hidden_layer(self):
        domain = np.ones((5, 5, 12), bool)
        seeds = surface_partition_seeds(domain, [0]*3, [1]*3, [[2, 2, 2]], [7], 1.01)
        self.assertEqual(seeds[2, 2, 2], 7)
        self.assertEqual(seeds[2, 2, 8], 0)

    def test_propagation_cannot_cross_hard_interface(self):
        domain = np.ones((7, 3, 3), bool)
        seeds = np.zeros(domain.shape, np.int32)
        seeds[0, 1, 1], seeds[6, 1, 1] = 1, 2
        blocked = np.zeros((3, *domain.shape), bool)
        blocked[0, 2] = True
        labels, confidence = propagate_partitions(domain, seeds, [1]*3, blocked_edges=blocked)
        self.assertTrue(np.all(labels[:3] == 1))
        self.assertTrue(np.all(labels[3:] == 2))
        self.assertTrue(np.isfinite(confidence).all())
        seeds[6, 1, 1] = 0
        with self.assertRaisesRegex(ValueError, "Unseeded"):
            propagate_partitions(domain, seeds, [1]*3, blocked_edges=blocked)


class SegmentGuardTests(unittest.TestCase):
    def test_same_end_labels_do_not_hide_thin_obstacle(self):
        labels = torch.ones((8, 3, 3), dtype=torch.long)
        labels[3] = 2
        a = torch.tensor([[1., 1., 1.]])
        b = torch.tensor([[6., 1., 1.]])
        roots = torch.tensor([1])
        hit = first_invalid_segment_fraction(a, b, roots, labels, [0]*3, [7,2,2])
        self.assertAlmostEqual(hit.item(), 0.3)
        corrected, invalid = constrain_volume_segments(a, b, roots, labels, [0]*3, [7,2,2])
        self.assertTrue(invalid.item())
        self.assertLess(corrected[0,0].item(), 2.5)
        self.assertTrue(torch.isinf(first_invalid_segment_fraction(a, corrected, roots, labels, [0]*3, [7,2,2])).all())

    def test_corner_contact_and_stationary_face_contact(self):
        labels = torch.ones((4,4,4), dtype=torch.long)
        labels[1,0,1] = 0
        a = torch.tensor([[0.,0.,1.], [.5,0.,1.]])
        b = torch.tensor([[2.,2.,1.], [.5,0.,2.]])
        hits = first_invalid_segment_fraction(a,b,torch.ones(2,dtype=torch.long),labels,[0]*3,[3]*3)
        self.assertAlmostEqual(hits[0].item(), .25)
        self.assertEqual(hits[1].item(), 0)

    def test_outside_box_is_not_border_padded(self):
        labels = torch.ones((3,3,3), dtype=torch.long)
        a = torch.tensor([[1.,1.,1.]])
        b = torch.tensor([[3.,1.,1.]])
        hits = first_invalid_segment_fraction(a,b,torch.tensor([1]),labels,[0]*3,[2]*3)
        self.assertAlmostEqual(hits.item(), .5)

    def test_export_audit_keeps_invalid_strand_identity(self):
        labels = np.ones((8,3,3), np.int32)
        labels[3] = 0
        strands = np.array([[[1.,1.,1.],[6.,1.,1.]],[[1.,1.,1.],[2.,1.,1.]]])
        report = audit_volume_strands(strands, [1,1], labels, [0]*3, [7,2,2])
        self.assertFalse(report["passed"])
        self.assertEqual(report["invalid_locations"], [[0,0]])
        self.assertEqual(report["strands"], 2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
