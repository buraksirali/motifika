#!/usr/bin/env python3
"""Renumber files sequentially."""
import os
from pathlib import Path

OUT_DIR = Path("/home/burakeda/Projeler/motifika/hayat_agaci")
files = sorted(OUT_DIR.glob("hayatagaci_*"))
print(f"Found {len(files)} files")

# Two-pass rename to avoid collision
tmp_names = []
for i, f in enumerate(files, 1):
    ext = f.suffix
    tmp = OUT_DIR / f".rename_{i:03d}{ext}"
    f.rename(tmp)
    tmp_names.append((tmp, ext))

for i, (tmp, ext) in enumerate(tmp_names, 1):
    final = OUT_DIR / f"hayatagaci_{i:03d}{ext}"
    tmp.rename(final)

print("Renumbered.")
for f in sorted(OUT_DIR.glob("hayatagaci_*")):
    print(f.name)
