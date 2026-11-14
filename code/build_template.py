"""Single-chain mmCIF self-template (table 1, row 'template +/- MSA', H4).

H4 asks whether the evolutionary channel's contribution is a *structural prior*
rather than irreducibly evolutionary information: given the target's own
structure as a template, does removing the MSA still cost anything?  Section
4.4 finds it does not (-0.308 without a template, +0.007 with one).

The template is the target chain of our own intact 'full' fold.  That is close
to a best-case prior and the circularity is stated as a limitation (section 6):
the direction should hold for any sufficient prior, the magnitude may not.

Two defects this script exists to work around (appendix A):
  * AF3's template parser cross-references _atom_site against
    _pdbx_poly_seq_scheme, so filtering only one of them makes it fail.  Both
    are filtered to the target chain here.
  * AF3 date-filters templates and predicted models carry no release date, so
    _pdbx_audit_revision_history.revision_date is injected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from af3_io import TARGET_CHAIN, fold_name, make_input, read_pipeline_msa, write_input
from build_excised import find_model_cif
from cif_utils import Item, Loop, parse_cif, read_residues, chain_sequence
from dataset import load_dataset
from msa_utils import empty_a3m

REVISION_DATE = "1900-01-01"   # earlier than any AF3 template cutoff


def single_chain_cif(cif_path: Path, chain: str = TARGET_CHAIN,
                     revision_date: str = REVISION_DATE) -> str:
    block = parse_cif(cif_path)

    atom = block.loop("_atom_site")
    if atom is None:
        raise ValueError(f"{cif_path}: no _atom_site loop")

    def chain_of(row: dict) -> str:
        return (row.get("_atom_site.auth_asym_id")
                or row.get("_atom_site.label_asym_id"))

    kept_entities = {r.get("_atom_site.label_entity_id")
                     for r in (dict(zip(atom.tags, row)) for row in atom.rows)
                     if chain_of(r) == chain}
    kept_entities.discard(None)

    atom.filter_rows(lambda r: chain_of(r) == chain)
    if not atom.rows:
        raise ValueError(f"{cif_path}: chain {chain!r} not found")

    # the cross-referenced category: filter it the same way, or the parse fails
    poly = block.loop("_pdbx_poly_seq_scheme")
    if poly is not None:
        poly.filter_rows(lambda r: (
            r.get("_pdbx_poly_seq_scheme.pdb_strand_id")
            or r.get("_pdbx_poly_seq_scheme.asym_id")) == chain)

    # keep the remaining chain-scoped categories consistent
    sa = block.loop("_struct_asym")
    if sa is not None:
        sa.filter_rows(lambda r: r.get("_struct_asym.id") == chain)
    for cat, tag in (("_entity", "_entity.id"),
                     ("_entity_poly", "_entity_poly.entity_id"),
                     ("_entity_poly_seq", "_entity_poly_seq.entity_id")):
        lp = block.loop(cat)
        if lp is not None and kept_entities:
            lp.filter_rows(lambda r, t=tag: r.get(t) in kept_entities)
    for cat in ("_struct_conn", "_pdbx_struct_assembly_gen", "_atom_type"):
        lp = block.loop(cat)
        if lp is not None and cat == "_struct_conn":
            block.drop_category(cat)

    # AF3 date-filters templates; a predicted model carries no release date
    block.drop_category("_pdbx_audit_revision_history")
    block.blocks.append(Loop(
        tags=["_pdbx_audit_revision_history.ordinal",
              "_pdbx_audit_revision_history.data_content_type",
              "_pdbx_audit_revision_history.major_revision",
              "_pdbx_audit_revision_history.minor_revision",
              "_pdbx_audit_revision_history.revision_date"],
        rows=[["1", "'Structure model'", "1", "0", revision_date]],
        styles=[["bare", "single", "bare", "bare", "bare"]],
    ))
    if not any(getattr(b, "tag", "") == "_entry.id" for b in block.blocks):
        block.blocks.insert(0, Item("_entry.id", block.name or "TMPL"))
    return block.render()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids_file", "--ids-file", dest="ids_file", default="ex_ids.txt")
    p.add_argument("--condition", choices=["full", "none"], required=True,
                   help="'none' = template supplied AND alignment removed")
    p.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    p.add_argument("--dataset", default="final_dataset.csv")
    p.add_argument("--targets-dir", default="targets")
    p.add_argument("--baseline-folds", default="e2_out_full")
    p.add_argument("--baseline-seed", type=int, default=1)
    p.add_argument("--template-dir", default="templates",
                   help="where the single-chain mmCIFs are cached")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    wanted = {ln.strip() for ln in Path(a.ids_file).read_text().splitlines()
              if ln.strip()}
    df = load_dataset(a.dataset)
    df = df[df.design_id.isin(wanted)]
    tdir = Path(a.template_dir); tdir.mkdir(parents=True, exist_ok=True)
    root = Path(a.baseline_folds)

    # one self-template per unique target sequence
    templates: dict[str, str] = {}
    for uid, g in df.groupby("target_uid"):
        cached = tdir / f"{uid}.cif"
        if cached.exists():
            templates[uid] = cached.read_text()
            continue
        src = None
        for did in g.design_id:
            src = find_model_cif(root, fold_name(did, "full", a.baseline_seed))
            if src is not None:
                break
        if src is None:
            raise FileNotFoundError(f"{uid}: no baseline fold under {root}")
        text = single_chain_cif(src)
        cached.write_text(text)
        templates[uid] = text
        res = read_residues(src, chain="A")
        print(f"[build_template] {g.target.iloc[0]:<16} template from {src.name} "
              f"({len(res)} residues) -> {cached}")

    cache: dict[str, tuple[str, str, str]] = {}
    n = 0
    for _, row in df.iterrows():
        uid = row.target_uid
        if uid not in cache:
            cache[uid] = read_pipeline_msa(Path(a.targets_dir) / uid)
        pipe_seq, unpaired, paired = cache[uid]
        L = len(row.target_seq)
        tmpl = [{
            "mmcif": templates[uid],
            # self-template: identity mapping between query and template
            "queryIndices": list(range(L)),
            "templateIndices": list(range(L)),
        }]
        if a.condition == "full":
            tgt_msa, tgt_paired = unpaired, paired
        else:
            tgt_msa, tgt_paired = empty_a3m(row.target_seq), ""

        cond = f"template_{a.condition}"
        obj = make_input(name=fold_name(row.design_id, cond, a.seed), seed=a.seed,
                         target_seq=row.target_seq, binder_seq=row.binder_seq,
                         target_msa=tgt_msa,
                         binder_msa=empty_a3m(row.binder_seq),
                         target_templates=tmpl)
        obj["sequences"][0]["protein"]["pairedMsa"] = tgt_paired
        write_input(obj, a.out_dir)
        n += 1

    print(f"[build_template] condition=template_{a.condition} seed={a.seed}: "
          f"{n} folds -> {a.out_dir}/")


if __name__ == "__main__":
    main()
