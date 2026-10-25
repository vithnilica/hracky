"""Fake 'laser crystal' preview: additive point projection, white on black.

usage: python preview.py OUTDIR  (reads ribosome_points.xyz, writes preview_*.png)
"""
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1]
pts = np.loadtxt(f"{d}/ribosome_points.xyz")


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


views = {"front": 0, "side": 90, "angle": 35}
fig, axs = plt.subplots(1, 3, figsize=(18, 7), facecolor="black")
for ax, (name, deg) in zip(axs, views.items()):
    p = pts @ rot_z(np.radians(deg)).T
    h, _, _ = np.histogram2d(p[:, 2], p[:, 0], bins=700, range=[[-35, 35], [-35, 35]])
    img = 1 - np.exp(-h / 3.0)  # saturating like light scattered from micro-cracks
    ax.imshow(img, origin="lower", cmap="bone", extent=[-35, 35, -35, 35], vmin=0, vmax=1)
    ax.set_title(name, color="white")
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{d}/preview.png", dpi=90, facecolor="black")
