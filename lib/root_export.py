"""Conservative root export adapter preserving connector and growth provenance."""
from __future__ import annotations

import numpy as np

from lib.recon_strategy.volume_root_connection import audit_root_connections


def export_root_set(plan, labels, solid, evidence, conflict, low, high):
    audit = audit_root_connections(plan, labels, solid, evidence, conflict, low, high)
    if not audit["passed"]:
        raise ValueError("root connector audit failed before export")
    statuses = np.asarray(plan["status"]).astype(str)
    participating = plan["root_labels"] > 0
    records = {
        "roots_world": np.asarray(plan["roots_world"]),
        "starts_world": np.asarray(plan["starts_world"]),
        "root_labels": np.asarray(plan["root_labels"]),
        "status": statuses,
        "participating": participating,
        "initial_connector": statuses == "short_connection",
        "growth_start_index": np.where(statuses == "short_connection", 1, 0).astype(np.int32),
        "audit": audit,
    }
    return records
