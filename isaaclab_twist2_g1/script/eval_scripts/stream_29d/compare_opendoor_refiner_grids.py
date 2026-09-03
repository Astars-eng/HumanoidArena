#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


KEYS = ("checkpoint", "horizon", "flow")


def load(path: Path) -> dict[tuple[int, int, int], dict]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        tuple(int(row[key]) for key in KEYS): row
        for row in rows
    }


def main() -> int:
    if len(sys.argv) not in (3, 4):
        raise SystemExit("usage: compare_opendoor_refiner_grids.py OFF_SUMMARY ON_SUMMARY [OUTPUT]")
    off_path, on_path = map(Path, sys.argv[1:3])
    output = Path(sys.argv[3]) if len(sys.argv) == 4 else on_path.parent / "refiner_on_vs_off.csv"
    off_rows, on_rows = load(off_path), load(on_path)
    rows = []
    for key in sorted(set(off_rows) | set(on_rows)):
        off, on = off_rows.get(key, {}), on_rows.get(key, {})
        off_rate = float(off.get("success_rate", 0) or 0)
        on_rate = float(on.get("success_rate", 0) or 0)
        rows.append({
            "checkpoint": key[0],
            "horizon": key[1],
            "flow": key[2],
            "off_completed": int(off.get("completed", 0) or 0),
            "off_successes": int(off.get("successes", 0) or 0),
            "off_success_rate": off_rate,
            "on_completed": int(on.get("completed", 0) or 0),
            "on_successes": int(on.get("successes", 0) or 0),
            "on_success_rate": on_rate,
            "success_rate_delta_on_minus_off": on_rate - off_rate,
            "status": "complete" if int(off.get("completed", 0) or 0) == 60 and int(on.get("completed", 0) or 0) == 60 else "pending",
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} ({sum(row['status'] == 'complete' for row in rows)}/{len(rows)} comparable pairs complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
