"""Intact-target conditions (table 1, rows 'full' and 'none').

    --condition full   parent alignment, intact target, no template  (baseline)
    --condition none   empty alignment, intact target, no template   (H1)

Exactly one channel changes between the two: the target's unpairedMsa is
replaced by a single-sequence alignment, so outer_product_mean has nothing to
contract and the pair representation receives no evolutionary term.  Binder
sequences are always the original and templates are disabled (section 3.3).

--seed sets both modelSeeds and the fold name.  The seed MUST be in the name:
without it, seeds 2 and 3 wrote to timestamp-disambiguated directories that
could not be attributed to a seed, and those folds had to be discarded and
rerun (appendix A).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from af3_io import make_input, read_pipeline_msa, write_input, fold_name
from dataset import load_dataset
from msa_utils import empty_a3m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids_file", "--ids-file", dest="ids_file", default="e2_ids.txt")
    p.add_argument("--condition", choices=["full", "none"], required=True)
    p.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    p.add_argument("--dataset", default="final_dataset.csv")
    p.add_argument("--targets-dir", default="targets")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--keep-paired-msa", action="store_true",
                   help="keep the paired MSA in the 'full' arm (default: keep)")
    a = p.parse_args()

    wanted = {ln.strip() for ln in Path(a.ids_file).read_text().splitlines()
              if ln.strip()}
    df = load_dataset(a.dataset)
    df = df[df.design_id.isin(wanted)]
    missing = wanted - set(df.design_id)
    if missing:
        print(f"[build_inputs] WARNING {len(missing)} ids not in dataset, "
              f"e.g. {sorted(missing)[:3]}")

    # one alignment read per unique target sequence, not per design
    cache: dict[str, tuple[str, str, str]] = {}
    n = 0
    for _, row in df.iterrows():
        uid = row.target_uid
        if uid not in cache:
            cache[uid] = read_pipeline_msa(Path(a.targets_dir) / uid)
        pipe_seq, unpaired, paired = cache[uid]
        if pipe_seq != row.target_seq:
            raise ValueError(
                f"{uid}: data-pipeline sequence differs from the dataset's "
                f"target sequence; alignment and target are out of register")

        if a.condition == "full":
            tgt_msa, tgt_paired = unpaired, (paired if a.keep_paired_msa else paired)
        else:                                   # the ablation
            tgt_msa, tgt_paired = empty_a3m(row.target_seq), ""

        name = fold_name(row.design_id, a.condition, a.seed)
        obj = make_input(name=name, seed=a.seed,
                         target_seq=row.target_seq, binder_seq=row.binder_seq,
                         target_msa=tgt_msa,
                         binder_msa=empty_a3m(row.binder_seq),
                         target_templates=[])          # templates disabled
        obj["sequences"][0]["protein"]["pairedMsa"] = tgt_paired
        write_input(obj, a.out_dir)
        n += 1

    print(f"[build_inputs] condition={a.condition} seed={a.seed}: "
          f"{n} folds -> {a.out_dir}/")


if __name__ == "__main__":
    main()
