"""AF3 input-JSON construction, fold naming, and output discovery.

Fold naming is deliberate: the seed is part of the name.  Until this was
corrected, build_inputs.py omitted it and seeds 2 and 3 wrote to
timestamp-disambiguated directories that could not be attributed to a seed
(appendix A, "defects worth knowing about"); those folds were discarded and
rerun.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

TARGET_CHAIN = "A"
BINDER_CHAIN = "B"

# Conditions of table 1.  'full' is the baseline: parent alignment, intact
# target, no template.
CONDITIONS = ("full", "none", "excised_full", "excised_none",
              "template_full", "template_none")


def fold_name(design_id: str, condition: str, seed: int, suffix: str = "") -> str:
    """AF3 lowercases job names and uses them as output directory names."""
    stem = f"{design_id}__{condition}__s{seed}"
    if suffix:
        stem += f"__{suffix}"
    return re.sub(r"[^a-z0-9_.-]", "_", stem.lower())


def parse_fold_name(name: str) -> dict | None:
    parts = name.split("__")
    if len(parts) < 3 or not parts[2].startswith("s"):
        return None
    out = {"design_id": parts[0], "condition": parts[1]}
    try:
        out["seed"] = int(parts[2][1:])
    except ValueError:
        return None
    out["suffix"] = "__".join(parts[3:]) if len(parts) > 3 else ""
    return out


def protein_chain(chain_id: str, sequence: str, unpaired_msa: str | None,
                  paired_msa: str = "", templates: list | None = None) -> dict:
    """One protein entry of an AF3 input JSON.

    unpaired_msa is given explicitly in every condition — never left unset —
    so AF3 never silently runs its own search at inference time.  An alignment
    containing only the query is the ablated channel.
    """
    entry = {
        "id": chain_id,
        "sequence": sequence,
        "unpairedMsa": unpaired_msa if unpaired_msa is not None else f">query\n{sequence}\n",
        "pairedMsa": paired_msa,
        "templates": templates if templates is not None else [],
    }
    return {"protein": entry}


def make_input(name: str, seed: int, target_seq: str, binder_seq: str,
               target_msa: str | None, binder_msa: str | None = None,
               target_templates: list | None = None) -> dict:
    """Two-chain input: A = target (the channel we ablate), B = binder.

    The binder's own alignment is empty by construction — designed binders have
    no natural relatives — so all alignment information comes from the target
    (section 1.1).
    """
    return {
        "name": name,
        "modelSeeds": [int(seed)],
        "sequences": [
            protein_chain(TARGET_CHAIN, target_seq, target_msa,
                          templates=target_templates),
            protein_chain(BINDER_CHAIN, binder_seq, binder_msa),
        ],
        "dialect": "alphafold3",
        "version": 2,
    }


def write_input(obj: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{obj['name']}.json"
    path.write_text(json.dumps(obj, indent=2))
    return path


# --- data-pipeline products ---------------------------------------------------

def read_pipeline_msa(target_dir: str | Path, chain_id: str = TARGET_CHAIN
                      ) -> tuple[str, str, str]:
    """Pull (sequence, unpairedMsa, pairedMsa) out of an AF3 *_data.json.

    run_alphafold.py --norun_inference writes this next to the job directory;
    it is the only product of the ~1.3 h MSA queue that we reuse.
    """
    target_dir = Path(target_dir)
    cands = sorted(target_dir.glob("*_data.json")) or sorted(target_dir.glob("*data.json"))
    if not cands:
        raise FileNotFoundError(f"no *_data.json under {target_dir}")
    data = json.loads(cands[0].read_text())
    for s in data["sequences"]:
        prot = s.get("protein")
        if prot and prot.get("id") in (chain_id, [chain_id]):
            return prot["sequence"], prot.get("unpairedMsa", ""), prot.get("pairedMsa", "")
    prot = data["sequences"][0]["protein"]
    return prot["sequence"], prot.get("unpairedMsa", ""), prot.get("pairedMsa", "")


# --- output discovery ---------------------------------------------------------

def find_confidences(root: str | Path) -> "list[tuple[Path, Path | None]]":
    """Every (confidences.json, summary_confidences.json) pair under `root`."""
    root = Path(root)
    out = []
    for conf in root.rglob("*_confidences.json"):
        if "summary" in conf.name:
            continue
        summary = conf.with_name(conf.name.replace("_confidences.json",
                                                   "_summary_confidences.json"))
        out.append((conf, summary if summary.exists() else None))
    return sorted(out)


def is_complete(fold_dir: str | Path) -> bool:
    """Fold runners skip any fold that already has a _confidences.json, so an
    interrupted run resumes by re-invoking the same script (appendix D)."""
    p = Path(fold_dir)
    return p.is_dir() and any(
        f.name.endswith("_confidences.json") and "summary" not in f.name
        for f in p.iterdir()
    )
