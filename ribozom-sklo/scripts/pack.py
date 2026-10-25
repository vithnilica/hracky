"""Pack points + labels into int16 xyz (0.002 mm units) followed by uint8 labels."""
import sys
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
p = np.loadtxt(f"{src}/ribosome_points.xyz")
lab = np.load(f"{src}/labels.npy").astype(np.uint8)
q = np.round(p / 0.002).astype("<i2")
with open(dst, "wb") as fh:
    fh.write(np.uint32(len(p)).tobytes())
    fh.write(q.tobytes()); fh.write(lab.tobytes())
print(dst, len(p))
