"""Write text-free `<name>.metrics.json` siblings of result files that contain passage text.

Result JSONs from ALGEN, Vec2Text, GEIA and the transfer attack store per-passage source texts and
reconstructions (MS MARCO / NQ passages). For public release only the metrics are kept: this script removes those
keys and writes `<name>.metrics.json` next to each original; .gitignore excludes the originals.
Idempotent. Usage: python experiments/strip_text_from_results.py
"""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
TEXT_KEYS = {"reconstructions", "examples", "predictions", "source", "text", "texts"}


def has_text(o) -> bool:
    if isinstance(o, dict):
        return any(k in TEXT_KEYS for k in o) or any(has_text(v) for v in o.values())
    if isinstance(o, list):
        return any(has_text(x) for x in o[:3])
    return False


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in TEXT_KEYS}
    if isinstance(o, list):
        return [strip(x) for x in o]
    return o


def main() -> None:
    n = 0
    for f in sorted(RESULTS.rglob("*.json")):
        if f.name.endswith(".metrics.json"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if has_text(d):
            (f.with_suffix("").with_name(f.stem + ".metrics.json")).write_text(json.dumps(strip(d), indent=2), encoding="utf-8")
            n += 1
    print("wrote", n, "metrics-only files")


if __name__ == "__main__":
    main()
