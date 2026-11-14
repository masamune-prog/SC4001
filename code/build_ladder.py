"""Depth-matched random and diverse alignment subsets (table 1, H2 and H3).

The ladder is Neff80 in {1, 2, 5, 10, 30} crossed with {random, diverse}, on
*excised* targets, plus the full alignment as the top rung.

  H3 (interior optimum): does a thinned alignment beat the full one?  Section
     4.5 rejects it -- no rung beats full depth.
  H2 (depth over content): at matched effective depth, does it matter *what*
     is in the alignment?  Section 4.5 supports it -- random and diverse are
     indistinguishable, so depth dominates content.

The random arm is the control that matters: prior subsampling work shows that
apparent gains from selective sequence choice vanish once effective depth is
held fixed, so 'diverse' is only meaningful against a depth-matched 'random'.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from af3_io import fold_name, make_input, read_pipeline_msa, write_input
from dataset import load_dataset
from msa_utils import crop_a3m, empty_a3m, neff, read_a3m, subsample_to_neff

RUNGS = [1, 2, 5, 10, 30]
MODES = ["random", "diverse"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    p.add_argument("--ids_file", "--ids-file", dest="ids_file", default="ex_ids.txt")
    p.add_argument("--dataset", default="final_dataset.csv")
    p.add_argument("--targets-dir", default="targets")
    p.add_argument("--windows", default="excision_windows.json",
                   help="written by build_excised.py; the ladder is run on "
                        "excised targets")
    p.add_argument("--rungs", default=",".join(map(str, RUNGS)))
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--include-full", action="store_true", default=True,
                   help="also emit the full-depth rung for the 'vs. full' column")
    p.add_argument("--seed", type=int, default=1,
                   help="AF3 model seed; also seeds the subsampling draw")
    p.add_argument("--report", default="ladder_neff.tsv")
    a = p.parse_args()

    rungs = [int(x) for x in a.rungs.split(",") if x]
    modes = [m for m in a.modes.split(",") if m]
    windows = json.loads(Path(a.windows).read_text()) if Path(a.windows).exists() else {}
    if not windows:
        print("[build_ladder] WARNING no excision windows found; "
              "building the ladder on intact targets instead")

    wanted = {ln.strip() for ln in Path(a.ids_file).read_text().splitlines()
              if ln.strip()}
    df = load_dataset(a.dataset)
    df = df[df.design_id.isin(wanted)]

    # --- build one subsampled alignment per (target, rung, mode) ------------
    per_target: dict[str, dict] = {}
    rows = []
    for uid, g in df.groupby("target_uid"):
        seq, unpaired, _ = read_pipeline_msa(Path(a.targets_dir) / uid)
        w = windows.get(uid)
        if w:
            seq = seq[w["start"]:w["end"]]
            unpaired = crop_a3m(unpaired, w["start"], w["end"]) if unpaired.strip() \
                else empty_a3m(seq)
        variants = {}
        for rung in rungs:
            for mode in modes:
                sub = subsample_to_neff(unpaired, rung, mode=mode, seed=a.seed)
                achieved = neff(read_a3m(sub)[1])
                variants[f"neff{rung}_{mode}"] = sub
                rows.append((g.target.iloc[0], uid, rung, mode, achieved,
                             len(read_a3m(sub)[1])))
                print(f"[build_ladder] {g.target.iloc[0]:<16} rung={rung:<3} "
                      f"{mode:<8} achieved Neff80={achieved} "
                      f"({len(read_a3m(sub)[1])} rows)")
        if a.include_full:
            variants["full"] = unpaired
        per_target[uid] = {"seq": seq, "variants": variants}

    with open(a.report, "w") as fh:
        fh.write("target\ttarget_uid\trung\tmode\tachieved_neff80\tn_rows\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    # --- emit folds ----------------------------------------------------------
    n = 0
    for _, row in df.iterrows():
        d = per_target[row.target_uid]
        for key, msa in d["variants"].items():
            cond = f"ladder_{key}"
            obj = make_input(name=fold_name(row.design_id, cond, a.seed),
                             seed=a.seed, target_seq=d["seq"],
                             binder_seq=row.binder_seq, target_msa=msa,
                             binder_msa=empty_a3m(row.binder_seq),
                             target_templates=[])
            write_input(obj, a.out_dir)
            n += 1

    print(f"[build_ladder] {n} ladder inputs -> {a.out_dir}/  "
          f"(depths {rungs} x {modes}"
          f"{' + full' if a.include_full else ''})")
    print(f"[build_ladder] achieved depths -> {a.report}")


if __name__ == "__main__":
    main()
