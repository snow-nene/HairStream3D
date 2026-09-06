import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.multiview_partition_correspondence import correlate_partition_labels


def test_correspondence_records_ties_instead_of_using_label_order():
    result = correlate_partition_labels(
        [np.array([1, 2]), np.array([7, 8])],
        [np.array([[3, 4], [5, 6]]), np.array([[3, 4], [5, 6]])],
    )
    assert result["conflicts"]
    assert result["mapping"] == {}


def test_unique_overlap_gets_global_id():
    result = correlate_partition_labels(
        [np.array([1]), np.array([1])],
        [np.array([[3, 4]]), np.array([[3, 4]])],
    )
    assert len(result["mapping"]) == 1
