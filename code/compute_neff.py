"""Neff80 for every target alignment -> neff80.tsv.

Alignment depth is measured with Neff80, the number of distinct sequences when
those sharing >= 80% identity count as one (section 3.2).  This is the first
direct measurement of effective depth on this benchmark: section 4.2 uses it to
show that depth does not predict the ablation effect (rho = +0.055), and to
correct the source study's shallow/deep labels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from af3_io import read_pipeline_msa
from msa_utils import neff, read_a3m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets-dir", default="targets")
    p.add_argument("--table", default="targets.tsv")
    p.add_argument("--out", default="neff80.tsv")
    p.add_argument("--identity", type=float, default=0.80)
    p.add_argument("--max-rows", type=int, default=200_000)
    a = p.parse_args()

    names = {}
    if Path(a.table).exists():
        with open(a.table) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                names[row["target_uid"]] = row

    rows = []
    for d in sorted(Path(a.targets_dir).iterdir()):
        if not d.is_dir():
            continue
        try:
            seq, unpaired, paired = read_pipeline_msa(d)
        except FileNotFoundError:
            print(f"[compute_neff] skip {d.name}: data pipeline not run")
            continue
        _, seqs = read_a3m(unpaired) if unpaired.strip() else ([], [seq])
        n = neff(seqs, identity=a.identity, max_rows=a.max_rows)
        meta = names.get(d.name, {})
        rows.append({
            "target_uid": d.name,
            "target": meta.get("target", d.name),
            "length": len(seq),
            "n_seqs": len(seqs),
            "neff80": n,
        })
        print(f"[compute_neff] {rows[-1]['target']:<16} n_seqs={len(seqs):>7} "
              f"neff80={n}")

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target_uid", "target", "length",
                                           "n_seqs", "neff80"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"[compute_neff] -> {a.out}")


if __name__ == "__main__":
    main()
