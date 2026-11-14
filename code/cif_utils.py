"""Minimal mmCIF reader/writer (stdlib + numpy only).

Enough of the format to do two jobs:
  * read CA coordinates and chain/residue ids out of an AF3 model.cif
    (build_excised.py: the 11 A halo crop);
  * rewrite a model.cif as a single-chain template (build_template.py).

The file is kept as an ordered list of blocks so that anything we do not touch
is written back verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "SEC": "U", "PYL": "O", "MSE": "M",
}


# --- tokenizer ----------------------------------------------------------------

def _tokenize(text: str):
    """Yield (token, was_quoted) pairs, honouring ';' multi-line blocks."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):
            buf = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                buf.append(lines[i])
                i += 1
            i += 1
            yield ("\n".join(buf), "semicolon")
            continue
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        for m in re.finditer(r"'([^']*)'|\"([^\"]*)\"|(\S+)", line):
            if m.group(1) is not None:
                yield (m.group(1), "single")
            elif m.group(2) is not None:
                yield (m.group(2), "double")
            else:
                yield (m.group(3), "bare")
        i += 1


def _quote(value: str, style: str = "bare") -> str:
    if value == "":
        return "."
    if style == "semicolon" or "\n" in value:
        return f"\n;{value}\n;\n"
    if re.search(r"[\s'\"]", value) or value.startswith("_"):
        if "'" not in value:
            return f"'{value}'"
        return f'"{value}"'
    return value


# --- document model -----------------------------------------------------------

@dataclass
class Loop:
    tags: list[str]
    rows: list[list[str]]
    styles: list[list[str]] = field(default_factory=list)

    @property
    def category(self) -> str:
        return self.tags[0].split(".")[0]

    def column(self, tag: str) -> list[str]:
        full = tag if "." in tag else f"{self.category}.{tag}"
        j = self.tags.index(full)
        return [r[j] for r in self.rows]

    def filter_rows(self, predicate) -> None:
        keep = [i for i, r in enumerate(self.rows)
                if predicate(dict(zip(self.tags, r)))]
        self.rows = [self.rows[i] for i in keep]
        if self.styles:
            self.styles = [self.styles[i] for i in keep]

    def render(self) -> str:
        out = ["loop_"]
        out += [t for t in self.tags]
        for ri, row in enumerate(self.rows):
            styles = self.styles[ri] if self.styles else ["bare"] * len(row)
            out.append(" ".join(_quote(v, s) for v, s in zip(row, styles)))
        out.append("#")
        return "\n".join(out)


@dataclass
class Item:
    tag: str
    value: str
    style: str = "bare"

    @property
    def category(self) -> str:
        return self.tag.split(".")[0]

    def render(self) -> str:
        return f"{self.tag} {_quote(self.value, self.style)}"


@dataclass
class DataBlock:
    name: str
    blocks: list = field(default_factory=list)

    def loops(self, category: str) -> list[Loop]:
        return [b for b in self.blocks
                if isinstance(b, Loop) and b.category == category]

    def loop(self, category: str) -> Loop | None:
        ls = self.loops(category)
        return ls[0] if ls else None

    def drop_category(self, category: str) -> None:
        self.blocks = [b for b in self.blocks if
                       not (isinstance(b, (Loop, Item)) and b.category == category)]

    def render(self) -> str:
        parts = [f"data_{self.name}", "#"]
        parts += [b.render() for b in self.blocks]
        return "\n".join(parts) + "\n"


def parse_cif(path_or_text: str | Path) -> DataBlock:
    text = (Path(path_or_text).read_text()
            if (isinstance(path_or_text, Path) or
                ("\n" not in str(path_or_text) and Path(str(path_or_text)).exists()))
            else str(path_or_text))
    toks = list(_tokenize(text))
    block = DataBlock(name="unknown")
    i = 0
    while i < len(toks):
        tok, style = toks[i]
        if tok.lower().startswith("data_") and style == "bare":
            block.name = tok[5:]
            i += 1
        elif tok.lower() == "loop_" and style == "bare":
            i += 1
            tags = []
            while i < len(toks) and toks[i][0].startswith("_") and toks[i][1] == "bare":
                tags.append(toks[i][0])
                i += 1
            rows, styles = [], []
            n = len(tags)
            while i < len(toks):
                t, s = toks[i]
                if s == "bare" and (t.startswith("_") or t.lower() in ("loop_",) or
                                    t.lower().startswith("data_")):
                    break
                row = [t]; rowstyle = [s]
                i += 1
                while len(row) < n and i < len(toks):
                    row.append(toks[i][0]); rowstyle.append(toks[i][1]); i += 1
                if len(row) == n:
                    rows.append(row); styles.append(rowstyle)
            block.blocks.append(Loop(tags=tags, rows=rows, styles=styles))
        elif tok.startswith("_"):
            tag = tok
            if i + 1 < len(toks):
                block.blocks.append(Item(tag, toks[i + 1][0], toks[i + 1][1]))
                i += 2
            else:
                i += 1
        else:
            i += 1
    return block


# --- coordinates --------------------------------------------------------------

@dataclass
class Residue:
    chain: str
    seq_id: int
    comp_id: str
    coords: np.ndarray            # (n_atoms, 3)
    ca: np.ndarray | None

    @property
    def one_letter(self) -> str:
        return THREE_TO_ONE.get(self.comp_id.upper(), "X")


def read_residues(path: str | Path, chain: str | None = None,
                  use_auth: bool = True) -> list[Residue]:
    block = parse_cif(Path(path))
    loop = block.loop("_atom_site")
    if loop is None:
        raise ValueError(f"{path}: no _atom_site loop")
    tags = loop.tags
    def col(*names):
        for n in names:
            full = f"_atom_site.{n}"
            if full in tags:
                return tags.index(full)
        raise KeyError(names)

    ci = col("auth_asym_id", "label_asym_id") if use_auth else col("label_asym_id")
    si = col("label_seq_id", "auth_seq_id")
    gi = col("label_comp_id", "auth_comp_id")
    ai = col("label_atom_id", "auth_atom_id")
    xi, yi, zi = col("Cartn_x"), col("Cartn_y"), col("Cartn_z")

    acc: dict[tuple[str, int], dict] = {}
    order: list[tuple[str, int]] = []
    for r in loop.rows:
        ch = r[ci]
        if chain is not None and ch != chain:
            continue
        try:
            sid = int(r[si])
        except ValueError:
            continue
        key = (ch, sid)
        if key not in acc:
            acc[key] = {"comp": r[gi], "xyz": [], "ca": None}
            order.append(key)
        xyz = (float(r[xi]), float(r[yi]), float(r[zi]))
        acc[key]["xyz"].append(xyz)
        if r[ai].strip() == "CA":
            acc[key]["ca"] = xyz
    out = []
    for key in order:
        ch, sid = key
        d = acc[key]
        out.append(Residue(ch, sid, d["comp"], np.asarray(d["xyz"], dtype=float),
                           np.asarray(d["ca"], dtype=float) if d["ca"] else None))
    return out


def chain_sequence(residues: list[Residue]) -> str:
    return "".join(r.one_letter for r in residues)


def min_heavy_atom_distances(a: list[Residue], b: list[Residue]) -> np.ndarray:
    """For each residue of `a`, the distance to the nearest atom of `b`."""
    B = np.concatenate([r.coords for r in b], axis=0)
    out = np.empty(len(a))
    for i, r in enumerate(a):
        d = np.linalg.norm(r.coords[:, None, :] - B[None, :, :], axis=-1)
        out[i] = d.min()
    return out
