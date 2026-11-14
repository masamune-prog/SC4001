"""11 A halo crop plus the matched alignment column crop (table 1, R1).

Epitope excision, as introduced by Kim et al.: the target is cropped to the
shortest contiguous window covering every residue within 11 A of the binder,
and the same columns are removed from the alignment.  No alignment *rows* are
discarded (section 3.4) -- only columns -- so the cropped a3m stays aligned to
the cropped query.

The halo is read from the corresponding *baseline* fold (condition 'full'),
which is why this script runs after run_e2.sh.  Contacts are taken over all
heavy atoms, not CA, and the window is formed per target as the union over that
target's designs so that one alignment crop serves every fold of that target.

Completing the intact--MSA cell of the 2x2 (table 2) is the point of this arm:
the source benchmark never ran it, which is why the interaction between MSA and
excision is unidentifiable from their data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from af3_io import (BINDER_CHAIN, TARGET_CHAIN, fold_name, make_input,
                    read_pipeline_msa, write_input)
from cif_utils import min_heavy_atom_distances, read_residues
from dataset import load_dataset
from msa_utils import crop_a3m, empty_a3m


def find_model_cif(fold_root: Path, name: str) -> Path | None:
    d = fold_root / name
    if not d.is_dir():
        cands = list(fold_root.glob(f"{name}*"))
        d = cands[0] if cands else None
    if d is None or not d.is_dir():
        return None
    for pat in ("*_model.cif", "*model*.cif", "*.cif"):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    return None


def halo_for_design(cif: Path, radius: float) -> np.ndarray:
    """Indices (0-based, along the target chain) within `radius` of the binder."""
    tgt = read_residues(cif, chain=TARGET_CHAIN)
    bnd = read_residues(cif, chain=BINDER_CHAIN)
    if not tgt or not bnd:
        raise ValueError(f"{cif}: missing chain {TARGET_CHAIN} or {BINDER_CHAIN}")
    d = min_heavy_atom_distances(tgt, bnd)
    return np.flatnonzero(d <= radius)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids_file", "--ids-file", dest="ids_file", default="ex_ids.txt")
    p.add_argument("--condition", choices=["full", "none"], required=True)
    p.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    p.add_argument("--dataset", default="final_dataset.csv")
    p.add_argument("--targets-dir", default="targets")
    p.add_argument("--baseline-folds", default="e2_out_full",
                   help="AF3 output root of the 'full' condition, seed 1")
    p.add_argument("--baseline-seed", type=int, default=1)
    p.add_argument("--radius", type=float, default=11.0)
    p.add_argument("--min-window", type=int, default=85,
                   help="Kim et al. crop to contiguous 85-130 residue windows")
    p.add_argument("--max-window", type=int, default=130)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--windows-out", default="excision_windows.json")
    a = p.parse_args()

    wanted = {ln.strip() for ln in Path(a.ids_file).read_text().splitlines()
              if ln.strip()}
    df = load_dataset(a.dataset)
    df = df[df.design_id.isin(wanted)]
    root = Path(a.baseline_folds)

    # --- pass 1: the window, per unique target sequence ----------------------
    windows: dict[str, tuple[int, int]] = {}
    coverage: dict[str, float] = {}
    for uid, g in df.groupby("target_uid"):
        hits = []
        for did in g.design_id:
            cif = find_model_cif(root, fold_name(did, "full", a.baseline_seed))
            if cif is None:
                continue
            try:
                hits.append(halo_for_design(cif, a.radius))
            except Exception as exc:                 # noqa: BLE001
                print(f"[build_excised] {did}: {exc}")
        if not hits:
            raise FileNotFoundError(
                f"{uid}: no baseline folds under {root}; run run_e2.sh first")
        idx = np.unique(np.concatenate(hits))
        L = len(g.target_seq.iloc[0])
        start, end = int(idx.min()), int(idx.max()) + 1      # shortest contiguous
        # pad symmetrically into the 85-130 band, clamped to the sequence
        if end - start < a.min_window:
            need = a.min_window - (end - start)
            start = max(0, start - need // 2)
            end = min(L, start + a.min_window)
            start = max(0, end - a.min_window)
        if end - start > a.max_window:
            centre = (start + end) // 2
            start = max(0, centre - a.max_window // 2)
            end = min(L, start + a.max_window)
        windows[uid] = (start, end)
        coverage[uid] = (end - start) / L
        print(f"[build_excised] {g.target.iloc[0]:<16} L={L:<5} "
              f"window=[{start},{end}) ({end-start} aa, "
              f"{coverage[uid]*100:.0f}% of sequence, {len(idx)} halo residues)")

    Path(a.windows_out).write_text(json.dumps(
        {uid: {"start": s, "end": e, "coverage": coverage[uid]}
         for uid, (s, e) in windows.items()}, indent=2))

    # sntx is the control: its 11 A window kept 100% of the sequence, so its
    # excision effect (-0.002) measures the noise floor of this arm.
    for uid, cov in coverage.items():
        if cov > 0.999:
            print(f"[build_excised] NOTE {uid} keeps the whole sequence "
                  f"-- this target is a control, not a crop")

    # --- pass 2: cropped inputs ---------------------------------------------
    cache: dict[str, tuple[str, str, str]] = {}
    n = 0
    for _, row in df.iterrows():
        uid = row.target_uid
        start, end = windows[uid]
        if uid not in cache:
            cache[uid] = read_pipeline_msa(Path(a.targets_dir) / uid)
        pipe_seq, unpaired, paired = cache[uid]
        cropped_seq = pipe_seq[start:end]

        if a.condition == "full":
            tgt_msa = crop_a3m(unpaired, start, end) if unpaired.strip() \
                else empty_a3m(cropped_seq)
            tgt_paired = crop_a3m(paired, start, end) if paired.strip() else ""
        else:
            tgt_msa, tgt_paired = empty_a3m(cropped_seq), ""

        cond = f"excised_{a.condition}"
        name = fold_name(row.design_id, cond, a.seed)
        obj = make_input(name=name, seed=a.seed, target_seq=cropped_seq,
                         binder_seq=row.binder_seq, target_msa=tgt_msa,
                         binder_msa=empty_a3m(row.binder_seq),
                         target_templates=[])
        obj["sequences"][0]["protein"]["pairedMsa"] = tgt_paired
        write_input(obj, a.out_dir)
        n += 1

    print(f"[build_excised] condition=excised_{a.condition} seed={a.seed}: "
          f"{n} folds -> {a.out_dir}/")


if __name__ == "__main__":
    main()
