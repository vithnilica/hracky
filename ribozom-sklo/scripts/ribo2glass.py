"""PDB ribosome -> meshes (STL) + point cloud (PLY/XYZ/DXF) for subsurface laser engraving.

usage: python ribo2glass.py STRUCTURE.cif.gz OUTDIR [--size-mm 60] [--cut]
"""
import argparse
import os

import gemmi
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

# chain groups for PDB 7N1P (E. coli 70S, PRE-C)
TUNNEL = {"uL4": ["LD"], "uL22": ["LV"], "uL23": ["LW"], "uL24": ["LX"]}
TRNA = ["Pt", "Pp"]  # P-site tRNA + nascent peptide
SKIP = {"Dt"}  # A-site tRNA, not shown in the figure

# point spacing on the final object (mm): denser = brighter in glass
SPACING = {"50S": 0.22, "30S": 0.30, "P-tRNA": 0.10,
           "uL4": 0.12, "uL22": 0.12, "uL23": 0.12, "uL24": 0.12}


def load(path):
    st = gemmi.read_structure(path)
    st.remove_hydrogens()
    groups = {k: [] for k in SPACING}
    tunnel_chain = {c: k for k, cs in TUNNEL.items() for c in cs}
    for ch in st[0]:
        if ch.name in SKIP:
            continue
        if ch.name in tunnel_chain:
            g = tunnel_chain[ch.name]
        elif ch.name in TRNA:
            g = "P-tRNA"
        elif ch.name in ("16", "mR") or ch.name.startswith("S"):
            g = "30S"
        else:
            g = "50S"
        for res in ch:
            for at in res:
                groups[g].append(at.pos.tolist())
    groups = {k: np.array(v) for k, v in groups.items()}
    # PTC = 3' end of P-tRNA (A76)
    pt = st[0]["Pt"]
    ptc = np.array([a.pos.tolist() for a in pt[len(pt) - 1]]).mean(0)
    return groups, ptc


def frame(groups, ptc):
    """z: PTC -> tunnel exit (uL23/uL24), x: towards 30S. Origin = PTC."""
    exit_ = np.vstack([groups["uL23"], groups["uL24"]]).mean(0)
    z = exit_ - ptc
    z /= np.linalg.norm(z)
    s = groups["30S"].mean(0) - ptc
    x = s - z * (s @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return {k: (v - ptc) @ R.T for k, v in groups.items()}


def surface(xyz, grid=1.0, sigma=1.6, level=0.35, faces=0):
    """Gaussian-blob molecular surface (Angstrom)."""
    lo = xyz.min(0) - 6
    idx = np.floor((xyz - lo) / grid).astype(int)
    shape = idx.max(0) + 8
    rho = np.zeros(shape, np.float32)
    np.add.at(rho, tuple(idx.T), 1.0)
    rho = gaussian_filter(rho, sigma / grid)
    rho /= gaussian_filter(np.pad(np.ones((1, 1, 1), np.float32), 8), sigma / grid).max()
    if sigma > 2:
        # pick the iso-level that encloses ~20 A^3 per heavy atom (molecular volume)
        n_vox = int(len(xyz) * 20.0 / grid**3)
        level = np.partition(rho.ravel(), -n_vox)[-n_vox]
    v, f, _, _ = marching_cubes(rho, level, spacing=(grid,) * 3)
    m = trimesh.Trimesh(v + lo, f, process=True)
    if faces and len(m.faces) > faces:
        m = m.simplify_quadric_decimation(face_count=faces)
    # drop tiny detached blobs (stray ions/fragments) from smoothed envelopes
    if sigma > 2:
        parts = m.split(only_watertight=False)
        m = trimesh.util.concatenate([c for c in parts if c.area > 0.02 * m.area])
    m.fix_normals()
    return m


def write_dxf(path, pts):
    with open(path, "w") as fh:
        fh.write("0\nSECTION\n2\nENTITIES\n")
        for x, y, z in pts:
            fh.write(f"0\nPOINT\n8\n0\n10\n{x:.4f}\n20\n{y:.4f}\n30\n{z:.4f}\n")
        fh.write("0\nENDSEC\n0\nEOF\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cif")
    ap.add_argument("out")
    ap.add_argument("--size-mm", type=float, default=60.0, help="longest edge of the model")
    ap.add_argument("--cut", action="store_true", help="cut away front half of 50S to expose the tunnel")
    ap.add_argument("--density", type=float, default=1.0, help="multiplies point density")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="blur (A) for 50S/30S envelopes, e.g. 4-6 keeps only overall shape")
    ap.add_argument("--faces", type=int, default=0, help="decimate each part to this many triangles")
    ap.add_argument("--part-faces", type=int, default=0, help="triangles for each small part (default faces/4)")
    ap.add_argument("--part-smooth", type=float, default=1.6, help="blur (A) for tRNA and tunnel proteins")
    ap.add_argument("--single", action="store_true", help="export OBJ as one merged object")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    groups = frame(*load(a.cif))
    if a.cut:
        # remove 50S atoms in front of the tunnel plane (viewer at -y)
        g = groups["50S"]
        groups["50S"] = g[g[:, 1] > 0]

    meshes = {}
    for k, v in groups.items():
        if a.smooth and k in ("50S", "30S"):
            meshes[k] = surface(v, grid=2.0, sigma=a.smooth, level=0.5, faces=a.faces)
        else:
            meshes[k] = surface(v, sigma=a.part_smooth, faces=a.part_faces or a.faces // 4)
    allv = np.vstack([m.vertices for m in meshes.values()])
    center = (allv.min(0) + allv.max(0)) / 2
    scale = a.size_mm / (allv.max(0) - allv.min(0)).max()  # mm per Angstrom
    print(f"scale {scale:.3f} mm/A  ({scale * 10:.2f} mm/nm)")

    pts_all, labels = [], []
    for i, (k, m) in enumerate(meshes.items()):
        m.apply_translation(-center)
        m.apply_scale(scale)
        m.export(os.path.join(a.out, f"part_{k}.stl"))
        sp = SPACING[k] / np.sqrt(a.density)
        n = int(m.area / sp**2)
        p, _ = trimesh.sample.sample_surface_even(m, n, radius=sp * 0.7, seed=1)
        print(f"{k:7s} area {m.area:9.0f} mm2  points {len(p):8d}")
        pts_all.append(p)
        labels += [i] * len(p)
    trimesh.util.concatenate(list(meshes.values())).export(os.path.join(a.out, "ribosome.stl"))
    if a.single:
        trimesh.util.concatenate(list(meshes.values())).export(os.path.join(a.out, "ribosome.obj"))
    else:
        trimesh.Scene({k: m for k, m in meshes.items()}).export(os.path.join(a.out, "ribosome.obj"))

    pts = np.vstack(pts_all)
    print(f"TOTAL points {len(pts)}, bbox mm {np.ptp(pts, 0).round(1)}")
    trimesh.PointCloud(pts).export(os.path.join(a.out, "ribosome_points.ply"))
    np.savetxt(os.path.join(a.out, "ribosome_points.xyz"), pts, fmt="%.4f")
    write_dxf(os.path.join(a.out, "ribosome_points.dxf"), pts)
    np.save(os.path.join(a.out, "labels.npy"), np.array(labels))


if __name__ == "__main__":
    main()
