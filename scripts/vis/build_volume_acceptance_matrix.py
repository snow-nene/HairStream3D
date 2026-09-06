#!/usr/bin/env python3
"""汇总已运行的 64³ 体积分区对照，保留失败门禁和诊断字段。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    records = []
    for run in args.runs:
        report_path = args.root / run / "report.json"
        if not report_path.is_file():
            records.append({"run_id": run, "status": "missing_report", "passed": False})
            continue
        report = json.loads(report_path.read_text())
        records.append({"run_id": run, "status": report.get("status"),
                        "full_domain_acceptance": report.get("full_domain_acceptance"),
                        "domain_voxels": report.get("domain_voxels", report.get("candidate_voxels")),
                        "solved_voxels": report.get("solved_voxels"),
                        "unseeded_voxels": report.get("unseeded_voxels"),
                        "pde": report.get("pde"),
                        "growth_audit": report.get("growth_audit"),
                        "coverage": report.get("coverage"),
                        "leakage": report.get("leakage"),
                        "passed": report.get("full_domain_acceptance") is True})
        item = records[-1]
        growth = item.get("growth_audit") or {}
        pde = item.get("pde") or {}
        item["quality_gates"] = {
            "no_cross_partition": growth.get("partition_crossing_segments", 0) == 0,
            "no_out_of_domain": growth.get("invalid_locations", []) == [],
            "no_entity_penetration": growth.get("entity_penetration", 0) == 0,
            "pde_converged": bool(pde.get("converged", False)) if pde else False,
            "coverage_recorded": item.get("coverage") is not None,
            "leakage_recorded": item.get("leakage") is not None,
        }
        item["quality_passed"] = all(item["quality_gates"].values())
    output = {"version": 1, "resolution": "64^3", "runs": records,
              "all_passed": bool(records) and all(item["passed"] for item in records),
              "all_quality_gates_passed": bool(records) and all(item["quality_passed"] for item in records),
              "policy": "failed gates remain diagnostic; no default switch"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({"runs": len(records), "all_passed": output["all_passed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
