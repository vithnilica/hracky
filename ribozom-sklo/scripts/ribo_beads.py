"""Bead-style ribosome for glass: one sphere per residue/nucleotide, like the VMD figure.

50S/30S: only the outer skin of beads is kept (no clutter inside the glass).
P-tRNA and tunnel proteins: all beads, denser points.

usage: python ribo_beads.py STRUCTURE.cif.gz OUTDIR [--size-mm 60] [--radius 3.2] [--density 1]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gemmi
import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes

from ribo2glass import SKIP, SPACING, TRNA, TUNNEL, write_dxf

BEAD_ATOMS = ("CA", "P")


def load_beads(path):
    st = gemmi.read_structure(path)
    groups = {k: [] for k in SPACING}
    tunnel_chain = {c: k for k, cs in TUNNEL.items() for c in cs}
    for ch in st[0]:
        if ch.name in SKIP:
            continue
        g = (tunnel_chain.get(ch.name)
             or ("P-tRNA" if ch.name in TRNA else None)
             or ("30S" if ch.name in ("16", "mR") or ch.name.startswith("S") else "50S"))
        for res in ch:
            at = next((res[n][0] for n in BEAD_ATOMS if res.find_atom(n, "*")), None)
            if at is None:  # ligands, odd residues: use the residue centroid
                at_pos = np.mean([a.pos.tolist() for a in res], 0)
            else:
                at_pos = at.pos.tolist()
            groups[g].append(at_pos)
    pt = st[0]["Pt"]
    ptc = np.array([a.pos.tolist() for a in pt[len(pt) - 1]]).mean(0)
    return {k: np.array(v) for k, v in groups.items()}, ptc


def bead_mesh(xyz, lo, shape, grid, radius):
    """Surface of the union of spheres via a distance transform."""
    mask = np.ones(shape, bool)
    idx = np.round((xyz - lo) / grid).astype(int)
    mask[tuple(idx.T)] = False
    dist = ndimage.distance_transform_edt(mask, sampling=grid)
    v, f, _, _ = marching_cubes(dist, radius, spacing=(grid,) * 3)
    m = trimesh.Trimesh(v + lo, f, process=True)
    m.fix_normals()
    return m


def main():
    from ribo2glass import frame

    ap = argparse.ArgumentParser()
    ap.add_argument("cif")
    ap.add_argument("out")
    ap.add_argument("--size-mm", type=float, default=60.0)
    ap.add_argument("--radius", type=float, default=3.2, help="bead radius (A)")
    ap.add_argument("--skin", type=float, default=7.0, help="depth (A) of the visible bead layer")
    ap.add_argument("--density", type=float, default=1.0)
    ap.add_argument("--grid", type=float, default=0.7)
    ap.add_argument("--shell-factor", type=float, default=1.0, help="multiplies point spacing on 50S/30S")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    groups = frame(*load_beads(a.cif))
    allb = np.vstack(list(groups.values()))
    lo = allb.min(0) - a.radius - 4
    shape = tuple(np.ceil((allb.max(0) + a.radius + 4 - lo) / a.grid).astype(int) + 1)

    # whole-ribosome envelope (closed with a probe) to measure how deep a point lies
    mask = np.ones(shape, bool)
    mask[tuple(np.round((allb - lo) / a.grid).astype(int).T)] = False
    solid = ndimage.distance_transform_edt(mask, sampling=a.grid) < a.radius
    solid = ndimage.binary_closing(solid, ndimage.generate_binary_structure(3, 1),
                                   iterations=int(4 / a.grid))
    solid = ndimage.binary_fill_holes(solid)
    depth = ndimage.distance_transform_edt(solid, sampling=a.grid)

    meshes = {k: bead_mesh(v, lo, shape, a.grid, a.radius) for k, v in groups.items()}
    allv = np.vstack([m.vertices for m in meshes.values()])
    center = (allv.min(0) + allv.max(0)) / 2
    scale = a.size_mm / np.ptp(allv, 0).max()
    print(f"scale {scale:.3f} mm/A")

    pts_all, labels = [], []
    for i, (k, m) in enumerate(meshes.items()):
        sp = SPACING[k] / np.sqrt(a.density) * (a.shell_factor if k in ("50S", "30S") else 1)
        n = int(m.area * scale**2 / sp**2)
        p, _ = trimesh.sample.sample_surface_even(m, n, radius=sp * 0.7 / scale, seed=1)
        if k in ("50S", "30S"):
            ijk = np.clip(np.round((p - lo) / a.grid).astype(int), 0, np.array(shape) - 1)
            p = p[depth[tuple(ijk.T)] < a.skin]
        p = (p - center) * scale
        print(f"{k:7s} points {len(p):8d}")
        pts_all.append(p)
        labels += [i] * len(p)
        m.apply_translation(-center)
        m.apply_scale(scale)

    pts = np.vstack(pts_all)
    print(f"TOTAL points {len(pts)}, bbox mm {np.ptp(pts, 0).round(1)}")
    trimesh.PointCloud(pts).export(os.path.join(a.out, "ribosome_points.ply"))
    np.savetxt(os.path.join(a.out, "ribosome_points.xyz"), pts, fmt="%.4f")
    write_dxf(os.path.join(a.out, "ribosome_points.dxf"), pts)
    np.save(os.path.join(a.out, "labels.npy"), np.array(labels))


if __name__ == "__main__":
    main()
