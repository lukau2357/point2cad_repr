"""Inspect ABCParts .h5 files to see available keys, shapes, and sample values."""

import h5py
import numpy as np
import sys
import os
import glob


def inspect(path):
    print(f"\n{'='*60}")
    print(f"{path}")
    print(f"{'='*60}")
    with h5py.File(path, "r") as f:    
        for k in sorted(f.keys()):
            ds = f[k]
            print(f"  {k:<15} shape={str(ds.shape):<15} dtype={ds.dtype}")
            arr = np.array(ds)
            if arr.ndim == 1:
                unique = np.unique(arr)
                if len(unique) <= 20:
                    print(f"    unique values: {unique}")
                else:
                    print(f"    unique count: {len(unique)}, "
                          f"range=[{arr.min()}, {arr.max()}]")
            elif arr.ndim == 2:
                print(f"    first row: {arr[0]}")
                print(f"    row range: [{arr.min(axis=0)}, {arr.max(axis=0)}]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        default_dir = "../../abc_parts/ABC_final"
        paths = sorted(glob.glob(os.path.join(default_dir, "*.h5")))[:10]

        if not paths:
            print(f"No .h5 files found in {default_dir}")
            print("Usage: python scripts/inspect_h5.py file1.h5 [file2.h5 ...]")
            sys.exit(1)
        print(f"Inspecting first 3 files from {default_dir}")

    for p in paths:
        inspect(p)
