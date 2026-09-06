import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.spatial_budget import active_blocks, cache_fingerprint


def test_active_blocks_keep_physical_extent():
    blocks = active_blocks(np.ones((5, 5, 5), bool), [0.01, .01, .02], block_size=4)
    assert len(blocks) == 8
    assert blocks[0]["physical_extent_m"] == [.04, .04, .08]


def test_cache_fingerprint_changes_with_spacing_or_domain():
    domain = np.ones((2, 2, 2), bool)
    a = cache_fingerprint(domain, [.1, .1, .1], {"solver": "cg"})
    b = cache_fingerprint(domain, [.2, .1, .1], {"solver": "cg"})
    assert a != b
