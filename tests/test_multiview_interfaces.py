import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.multiview_interfaces import validate_volume_interfaces


def test_entity_direction_is_suppressed_and_hair_requires_partition():
    shape = (2, 2, 2)
    direction = np.ones((3, *shape))
    semantic = np.ones(shape, dtype=np.int8)
    semantic[0, 0, 0] = 2
    partition = np.ones(shape, dtype=np.int32)
    result = validate_volume_interfaces(np.zeros(shape), direction, semantic, partition)
    assert np.all(result["direction"][:, 0, 0, 0] == 0)


def test_invalid_hair_partition_is_rejected():
    with pytest.raises(ValueError):
        validate_volume_interfaces(np.zeros((1, 1, 1)), np.zeros((3, 1, 1, 1)),
                                   np.ones((1, 1, 1)), np.zeros((1, 1, 1)))
