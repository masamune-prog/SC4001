"""a3m parsing, Neff80, column cropping and depth-matched subsampling.

Conventions
-----------
An a3m alignment has one query (first record).  Uppercase letters and '-' are
*match* columns, aligned to the query; lowercase letters are insertions relative
to the query and occupy no match column.  Everything here keeps that invariant,
because AF3's parser requires it.
"""

from __future__ import annotations

import numpy as np

GAP = ord("-")


# --- IO ----------------------------------------------------------------------

def read_a3m(text: str) -> tuple[list[str], list[str]]:
    heads, seqs, cur = [], [], []
    for line in text.splitlines():
        if not line:
            continue
        if line[0] == ">":
            if cur:
                seqs.append("".join(cur))
                cur = []
            heads.append(line)
        else:
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    if len(heads) != len(seqs):
        raise ValueError(f"malformed a3m: {len(heads)} headers, {len(seqs)} sequences")
    return heads, seqs


def write_a3m(heads: list[str], seqs: list[str]) -> str:
    return "".join(f"{h}\n{s}\n" for h, s in zip(heads, seqs))


def empty_a3m(sequence: str, name: str = "query") -> str:
    """The ablated channel: a single-sequence alignment.

    With only the query present, outer_product_mean contracts over nothing and
    the pair representation receives no evolutionary term (section 2.1).
    """
    return f">{name}\n{sequence}\n"


# --- match-column matrix ------------------------------------------------------

def match_columns(seq: str) -> str:
    """Drop insertions, leaving one character per query match column."""
    return "".join(c for c in seq if c == "-" or c.isupper())


def to_matrix(seqs: list[str], max_rows: int | None = None) -> np.ndarray:
    rows = seqs if max_rows is None else seqs[:max_rows]
    m = [match_columns(s) for s in rows]
    width = len(m[0])
    bad = [i for i, s in enumerate(m) if len(s) != width]
    if bad:
        raise ValueError(f"ragged alignment: rows {bad[:5]} differ from query width {width}")
    return np.frombuffer("".join(m).encode("ascii"), dtype=np.uint8).reshape(len(m), width)


# --- effective depth ----------------------------------------------------------

def neff(seqs: list[str], identity: float = 0.80, min_overlap: int = 10,
         max_rows: int | None = 200_000) -> int:
    """Neff80: number of clusters when sequences sharing >= 80% identity merge.

    Greedy single-pass clustering against cluster representatives, which is the
    hhfilter-style definition used in section 3.2.  Identity is computed over
    columns where both sequences are ungapped.
    """
    if len(seqs) <= 1:
        return len(seqs)
    mat = to_matrix(seqs, max_rows=max_rows)
    ungapped = mat != GAP
    reps_seq: list[np.ndarray] = []
    reps_ung: list[np.ndarray] = []
    for i in range(mat.shape[0]):
        s, u = mat[i], ungapped[i]
        if reps_seq:
            R = np.stack(reps_seq)
            RU = np.stack(reps_ung)
            both = RU & u                       # (C, L)
            denom = both.sum(1)
            match = ((R == s) & both).sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                ident = np.where(denom >= min_overlap, match / np.maximum(denom, 1), 0.0)
            if (ident >= identity).any():
                continue
        reps_seq.append(s)
        reps_ung.append(u)
    return len(reps_seq)


# --- column cropping (epitope excision) ---------------------------------------

def crop_columns(seq: str, start: int, end: int) -> str:
    """Keep match columns [start, end) of an a3m row, with their insertions.

    An insertion is kept when the match column it follows is kept, so the
    cropped alignment stays a valid a3m against the cropped query.  No rows are
    discarded (section 3.4): only columns are removed.
    """
    out, col = [], 0
    for ch in seq:
        if ch == "-" or ch.isupper():
            if start <= col < end:
                out.append(ch)
            col += 1
        else:  # lowercase insertion, belongs to the previous match column
            if start <= col - 1 < end:
                out.append(ch)
    return "".join(out)


def crop_a3m(text: str, start: int, end: int) -> str:
    heads, seqs = read_a3m(text)
    return write_a3m(heads, [crop_columns(s, start, end) for s in seqs])


# --- depth-matched subsampling (the depth ladder) -----------------------------

def _neff_of_indices(mat: np.ndarray, ung: np.ndarray, idx: list[int],
                     identity: float, min_overlap: int) -> int:
    reps: list[int] = []
    for i in idx:
        if reps:
            R, RU = mat[reps], ung[reps]
            both = RU & ung[i]
            denom = both.sum(1)
            match = ((R == mat[i]) & both).sum(1)
            ident = np.where(denom >= min_overlap, match / np.maximum(denom, 1), 0.0)
            if (ident >= identity).any():
                continue
        reps.append(i)
    return len(reps)


def subsample_to_neff(text: str, target_neff: int, mode: str = "random",
                      seed: int = 1, identity: float = 0.80,
                      min_overlap: int = 10, max_rows: int = 20_000) -> str:
    """Return an a3m whose Neff80 is as close as possible to `target_neff`.

    mode='random'  : sequences added in a random order (the control).
    mode='diverse' : greedy max-min selection, i.e. each new sequence is the one
                     least similar to everything already chosen.

    H2 (section 4.5) is the comparison between these two at matched depth.
    """
    heads, seqs = read_a3m(text)
    if target_neff <= 1 or len(seqs) <= 1:
        return write_a3m(heads[:1], seqs[:1])

    pool = list(range(1, min(len(seqs), max_rows)))   # 0 is the query, always kept
    mat = to_matrix(seqs, max_rows=max_rows)
    ung = mat != GAP
    rng = np.random.default_rng(seed)

    if mode == "random":
        rng.shuffle(pool)
        order = pool
    elif mode == "diverse":
        order = []
        remaining = set(pool)
        # similarity to the chosen set, initialised against the query
        def ident_to(i, j):
            both = ung[i] & ung[j]
            d = both.sum()
            if d < min_overlap:
                return 0.0
            return float(((mat[i] == mat[j]) & both).sum() / d)
        best_sim = {i: ident_to(i, 0) for i in remaining}
        while remaining:
            nxt = min(remaining, key=lambda i: (best_sim[i], i))
            order.append(nxt)
            remaining.discard(nxt)
            for i in remaining:
                s = ident_to(i, nxt)
                if s > best_sim[i]:
                    best_sim[i] = s
            if len(order) > 4 * target_neff + 200:
                break
    else:
        raise ValueError(f"unknown mode {mode!r}")

    chosen = [0]
    for i in order:
        chosen.append(i)
        if _neff_of_indices(mat, ung, chosen, identity, min_overlap) >= target_neff:
            break
    return write_a3m([heads[i] for i in chosen], [seqs[i] for i in chosen])
