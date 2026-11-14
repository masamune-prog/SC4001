"""Unique target sequences and their data-pipeline inputs.

One alignment is built per *unique target sequence* rather than per binder
(section 3.2): the 13 retained receptors map to 16 sequences because some have
several constructs.  Writing 16 single-chain jobs instead of 3,532 two-chain
ones is what makes the MSA queue take ~1.3 h instead of days.

Output
    targets/<target_uid>.json   single-chain AF3 job, for --norun_inference
    targets.tsv                 target_uid -> receptor name, length, n designs

Then:  ./run_msa_queue.sh      (AF3 data pipeline, 3 at a time)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import load_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="final_dataset.csv")
    p.add_argument("--out-dir", default="targets")
    p.add_argument("--table", default="targets.tsv")
    p.add_argument("--ids-out", default="e2_ids.txt",
                   help="design ids retained for Study A")
    p.add_argument("--keep-pmhc", action="store_true")
    a = p.parse_args()

    df = load_dataset(a.dataset, keep_pmhc=a.keep_pmhc)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for uid, g in df.groupby("target_uid", sort=True):
        seq = g.target_seq.iloc[0]
        name = g.target.iloc[0]
        # Single chain, no templates.  The data pipeline fills unpairedMsa /
        # pairedMsa; inference is not run here.
        job = {
            "name": uid,
            "modelSeeds": [1],
            "sequences": [{"protein": {"id": "A", "sequence": seq}}],
            "dialect": "alphafold3",
            "version": 2,
        }
        (out / f"{uid}.json").write_text(json.dumps(job, indent=2))
        rows.append((uid, name, len(seq), len(g), int(g.label.sum())))

    with open(a.table, "w") as fh:
        fh.write("target_uid\ttarget\tlength\tn_designs\tn_binders\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    Path(a.ids_out).write_text("\n".join(df.design_id.tolist()) + "\n")

    print(f"[prepare_targets] {len(rows)} unique target sequences -> {out}/")
    print(f"[prepare_targets] {len(df)} design ids -> {a.ids_out}")
    print(f"[prepare_targets] next: ./run_msa_queue.sh")


if __name__ == "__main__":
    main()
