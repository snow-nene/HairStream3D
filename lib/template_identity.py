"""Bind generated strands to the evaluated Blender render template."""

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_template_identity(head_npz: Path) -> dict:
    report_path = head_npz.with_name("report.json")
    if not report_path.is_file():
        raise ValueError(f"missing evaluated template report: {report_path}")
    report = json.loads(report_path.read_text())
    template_path = Path(report.get("template", ""))
    expected = report.get("sha256")
    if not template_path.is_file() or not expected:
        raise ValueError("template report lacks an existing template and sha256")
    actual = sha256_file(template_path)
    if actual != expected:
        raise ValueError("template blend changed after evaluated head export")
    return {
        "template_path": str(template_path.resolve()),
        "template_sha256": actual,
        "template_head_npz": str(head_npz.resolve()),
        "template_head_npz_sha256": sha256_file(head_npz),
        "template_object": report.get("object"),
        "template_modifiers": report.get("modifiers", []),
    }


def require_bound_template(data, template_blend: Path, allow_unbound: bool = False) -> dict:
    expected = data.get("template_blend_sha256")
    actual = sha256_file(template_blend)
    if expected is None:
        if allow_unbound:
            return {"bound": False, "template_sha256": actual}
        raise ValueError("strand file has no evaluated Blender template identity")
    expected_text = str(expected.item() if hasattr(expected, "item") else expected)
    if expected_text != actual:
        raise ValueError("strand file template identity does not match render template")
    return {"bound": True, "template_sha256": actual}
