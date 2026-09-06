"""Partition-local observation lookup; never borrow anchors across a cut."""
import numpy as np
from scipy.spatial import cKDTree

from lib.recon_strategy.weighted_poisson import label_partition_components


def nearest_component_sources(domain, partitions, sources, spacing):
    """Return nearest source indices within each connected labelled component.

    Components without sources are rejected. The result is intended only for
    initialization / reference assignment; it does not replace the PDE solve.
    """
    components, count = label_partition_components(domain, partitions)
    indices = np.zeros((3,*domain.shape),np.int64)
    for component in range(1,count+1):
        mask=components==component
        targets=np.argwhere(mask)
        anchors=np.argwhere(mask & sources)
        if not len(anchors):
            raise ValueError(f"Partition component {component} has no local direction source")
        _,nearest=cKDTree(anchors*np.asarray(spacing)).query(targets*np.asarray(spacing))
        indices[:,mask]=anchors[nearest].T
    return indices
