#!/usr/bin/env python3
"""输出统一可见投影评价指标。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from lib.visibility_metrics import visible_projection_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="npz with uv/predicted/target/visible/hair_mask")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angle-threshold-deg", type=float, default=30.0)
    args = parser.parse_args()
    with np.load(args.input) as data:
        required = ("uv", "predicted_directions", "target_directions", "visible", "hair_mask")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"missing metric arrays: {missing}")
        report = visible_projection_metrics(data["uv"], data["predicted_directions"],
                                            data["target_directions"], data["visible"],
                                            data["hair_mask"], angle_threshold_deg=args.angle_threshold_deg)
    report["input"] = str(args.input.resolve())
    report["angle_threshold_deg"] = float(args.angle_threshold_deg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
