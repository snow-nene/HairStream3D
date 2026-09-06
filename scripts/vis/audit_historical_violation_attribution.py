#!/usr/bin/env python3
"""为历史违规发丝生成不猜测来源的逐根归因报告。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    point_ids = audit.get("point_label_violation_strands")
    indices = sorted(int(item) for item in point_ids) if point_ids and isinstance(point_ids, list) else sorted({int(item[0]) for item in audit.get("invalid_locations", [])})
    records = [{"strand_id": index, "invalid_samples": sum(int(item[0]) == index for item in audit.get("invalid_locations", [])),
                "integration": "unknown", "correction": "unknown", "resampling": "unknown", "export": "observed_final_ply",
                "evidence_status": "insufficient_intermediate_snapshots"} for index in indices]
    report = {"version": 1, "source": str(args.audit.resolve()), "violating_strands": records,
              "attribution_complete": False,
              "reason": "历史产物缺少积分/修正/重采样中间快照，不能从最终 PLY 反推来源"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"violating_strands": len(records), "attribution_complete": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
