"""Stage snapshot persistence for reproducible integration attribution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np


def save_stage_snapshot(output_dir, stage, arrays, metadata=None):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    path = output / f"{stage}.npz"
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {"stage": str(stage), "path": str(path.resolve()), "sha256": digest,
              "arrays": {key: list(np.asarray(value).shape) for key, value in arrays.items()},
              "metadata": metadata or {}}
    manifest = output / "stage_manifest.json"
    history = json.loads(manifest.read_text()) if manifest.exists() else {"version": 1, "stages": []}
    history["stages"].append(record); manifest.write_text(json.dumps(history, ensure_ascii=False, indent=2))
    return record
