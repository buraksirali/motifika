#!/usr/bin/env python3
"""Download tree of life kilim motif images, verify they meet criteria."""
import os
import sys
import hashlib
import time
import urllib.parse
from pathlib import Path

import requests
from PIL import Image

OUT_DIR = Path("/home/burakeda/Projeler/motifika/hayat_agaci")
OUT_DIR.mkdir(parents=True, exist_ok=True)

UA = (
    "MotifikaTrainingDataset/1.0 "
    "(educational research; buraksirali.dev@gmail.com)"
)

HEADERS = {
    "User-Agent": UA,
    "Accept": "image/avif,image/webp,image/png,image/jpeg,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
    "Referer": "https://www.bing.com/",
}


def existing_hashes():
    seen = set()
    for p in OUT_DIR.glob("*"):
        if p.is_file():
            try:
                seen.add(hashlib.md5(p.read_bytes()).hexdigest())
            except Exception:
                pass
    return seen


def next_index():
    nums = []
    for p in OUT_DIR.glob("hayatagaci_*.*"):
        try:
            nums.append(int(p.stem.split("_")[1]))
        except Exception:
            pass
    return max(nums) + 1 if nums else 1


def verify_and_save(content, url, idx, seen_hashes):
    """Validate the bytes are an image meeting criteria. Returns (success, reason)."""
    h = hashlib.md5(content).hexdigest()
    if h in seen_hashes:
        return False, f"duplicate hash"

    # Save to a temp path first, then verify with PIL
    tmp = OUT_DIR / f".tmp_{idx}"
    tmp.write_bytes(content)

    try:
        img = Image.open(tmp)
        img.verify()
        img = Image.open(tmp)  # re-open after verify
        fmt = (img.format or "").upper()
        w, h_dim = img.size
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return False, f"not an image: {e}"

    if fmt not in {"JPEG", "PNG", "WEBP", "GIF"}:
        tmp.unlink(missing_ok=True)
        return False, f"unsupported format: {fmt}"

    if w < 300 or h_dim < 300:
        tmp.unlink(missing_ok=True)
        return False, f"too small: {w}x{h_dim}"

    # extension
    ext_map = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif"}
    ext = ext_map[fmt]

    final = OUT_DIR / f"hayatagaci_{idx:03d}.{ext}"
    tmp.rename(final)
    seen_hashes.add(h)
    return True, f"saved {final.name} ({w}x{h_dim} {fmt})"


def download(url, idx, seen_hashes, timeout=20):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            return False, f"http {r.status_code}"
        ctype = r.headers.get("Content-Type", "").lower()
        if "html" in ctype:
            return False, f"html response"
        return verify_and_save(r.content, url, idx, seen_hashes)
    except Exception as e:
        return False, f"error: {e}"


def main(urls_file):
    urls = []
    with open(urls_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    seen = existing_hashes()
    idx = next_index()
    saved = 0
    skipped = 0
    print(f"Starting at index {idx}, {len(urls)} URLs to try")

    for url in urls:
        ok, reason = download(url, idx, seen)
        if ok:
            print(f"OK  [{idx:03d}] {reason} <- {url[:80]}")
            idx += 1
            saved += 1
        else:
            print(f"SKIP      {reason} <- {url[:80]}")
            skipped += 1
        time.sleep(0.3)

    print(f"\nDone. saved={saved} skipped={skipped} total_now={idx-1}")


if __name__ == "__main__":
    main(sys.argv[1])
