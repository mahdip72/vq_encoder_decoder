#!/usr/bin/env python3
"""
Backbone-only secondary structure calling from PDB files with N, CA, C.

- Computes phi/psi using a consistent dihedral convention
- Skips phi/psi across chain/gap breaks via peptide-bond distance checks
- Assigns secondary structure with standard-ish Ramachandran windows
- Smooths runs (>=4 for helices, >=3 for strands) to reduce spurious singles
- Works on single files or directories (recursively for *.pdb)

Output: CSV with columns:
chain, resseq, icode, resname, phi, psi, ss  (ss in {H,E,C})
"""

import os, sys, glob, math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


# ---------- geometry helpers ----------
def vec(a, b): return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def norm(v): return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def dihedral(p1, p2, p3, p4):
    """
    Consistent convention:
      b0 = p2->p1 ; b1 = p2->p3 ; b2 = p3->p4
      v = cross(b0, b1n) ; w = cross(b2, b1n)
      angle = atan2( dot(cross(v,w), b1n), dot(v,w) )
    """
    b0 = vec(p2, p1)
    b1 = vec(p2, p3)
    b2 = vec(p3, p4)
    b1n = (b1[0] / (norm(b1) + 1e-9), b1[1] / (norm(b1) + 1e-9), b1[2] / (norm(b1) + 1e-9))
    v = cross(b0, b1n)
    w = cross(b2, b1n)
    x = dot(v, w)
    y = dot(cross(v, w), b1n)
    return math.degrees(math.atan2(y, x))


def bond_ok(a, b, maxdist=1.8, mindist=1.1):
    """Check peptide C–N geometry to avoid computing phi/psi across gaps."""
    dx = b[0] - a[0];
    dy = b[1] - a[1];
    dz = b[2] - a[2]
    d2 = dx * dx + dy * dy + dz * dz
    return (mindist ** 2) <= d2 <= (maxdist ** 2)


# ---------- PDB parsing ----------
def parse_backbone(pdb_path: str):
    """Return ordered chains: {chain_id: [(key, atoms{N,CA,C}, resname), ...]}"""
    atoms_by_res: Dict[Tuple[str, int, str], Dict[str, Tuple[float, float, float]]] = {}
    resname_by_res: Dict[Tuple[str, int, str], str] = {}
    with open(pdb_path) as f:
        for ln in f:
            if not ln.startswith("ATOM"):
                continue
            name = ln[12:16].strip()
            if name not in ("N", "CA", "C"):
                continue
            resname = ln[17:20].strip()
            chain = ln[21].strip() or "A"
            try:
                resseq = int(ln[22:26])
            except ValueError:
                continue
            icode = ln[26].strip()
            x = float(ln[30:38]);
            y = float(ln[38:46]);
            z = float(ln[46:54])
            key = (chain, resseq, icode)
            atoms_by_res.setdefault(key, {})[name] = (x, y, z)
            resname_by_res[key] = resname
    keys = sorted(atoms_by_res, key=lambda k: (k[0], k[1], k[2]))
    chains = defaultdict(list)
    for k in keys:
        chains[k[0]].append((k, atoms_by_res[k], resname_by_res[k]))
    return chains


# ---------- phi/psi & assignment ----------
def compute_phi_psi(chains):
    rows = []
    for chain, items in chains.items():
        n = len(items)
        for i, (key, atoms, resn) in enumerate(items):
            phi = psi = None
            if i > 0 and "C" in items[i - 1][1] and all(x in atoms for x in ("N", "CA", "C")):
                if bond_ok(items[i - 1][1]["C"], atoms["N"]):
                    phi = dihedral(items[i - 1][1]["C"], atoms["N"], atoms["CA"], atoms["C"])
            if i < n - 1 and "N" in items[i + 1][1] and all(x in atoms for x in ("N", "CA", "C")):
                if bond_ok(atoms["C"], items[i + 1][1]["N"]):
                    psi = dihedral(atoms["N"], atoms["CA"], atoms["C"], items[i + 1][1]["N"])
            rows.append([chain, key[1], key[2], resn, phi, psi])
    return rows


def assign(phi, psi):
    if phi is None or psi is None:
        return "C"
    # α-helix (canonical-ish)
    if -90 <= phi <= -30 and -70 <= psi <= -15:
        return "H"
    # β-strand (positive psi band)
    if -160 <= phi <= -80 and 90 <= psi <= 180:
        return "E"
    return "C"


def smooth(labels, minH=4, minE=3):
    lab = labels[:]
    n = len(lab);
    i = 0
    while i < n:
        j = i
        while j < n and lab[j] == lab[i]:
            j += 1
        L = lab[i];
        run = j - i
        need = minH if L == "H" else (minE if L == "E" else 1)
        if run < need:
            for k in range(i, j):
                lab[k] = "C"
        i = j
    return lab


# ---------- main ----------
def process_one(pdb_path: str):
    chains = parse_backbone(pdb_path)
    out_rows = []
    for ch, items in chains.items():
        rows = compute_phi_psi({ch: items})
        # per-chain raw labels
        raw = [assign(r[4], r[5]) for r in rows]
        lab = smooth(raw, minH=4, minE=3)
        for (chain, resseq, icode, resn, phi, psi), ss in zip(rows, lab):
            out_rows.append((chain, resseq, icode, resn, phi, psi, ss))
    cnt = Counter([r[6] for r in out_rows])
    return out_rows, cnt


def write_csv(rows, out_path):
    with open(out_path, "w") as w:
        w.write("chain,resseq,icode,resname,phi,psi,ss\n")
        for chain, resseq, icode, resn, phi, psi, ss in rows:
            phi_s = "" if phi is None else f"{phi:.3f}"
            psi_s = "" if psi is None else f"{psi:.3f}"
            w.write(f"{chain},{resseq},{icode},{resn},{phi_s},{psi_s},{ss}\n")


def collect_files(paths: List[str]) -> List[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, fnames in os.walk(p):
                for fn in fnames:
                    if fn.lower().endswith(".pdb"):
                        files.append(os.path.join(root, fn))
        else:
            files.append(p)
    return files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python assign_ss_from_backbone.py <file_or_dir> [more paths...]")
        sys.exit(1)
    files = collect_files(sys.argv[1:])
    for f in files:
        try:
            rows, cnt = process_one(f)
        except Exception as e:
            print(os.path.basename(f), "ERROR:", e)
            continue
        total = sum(cnt.values()) or 1
        print(os.path.basename(f),
              f"H={cnt.get('H', 0)} ({100 * cnt.get('H', 0) / total:.1f}%)",
              f"E={cnt.get('E', 0)} ({100 * cnt.get('E', 0) / total:.1f}%)",
              f"C={cnt.get('C', 0)} ({100 * cnt.get('C', 0) / total:.1f}%)")
        out_csv = os.path.splitext(f)[0] + "_ss.csv"
        write_csv(rows, out_csv)
