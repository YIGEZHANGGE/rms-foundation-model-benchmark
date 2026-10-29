#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download Whitfield Xenium data from Zenodo (processed_data.zip first, then sections)."""
import os
import time
import requests

ROOT = r"E:\rms_persister\data\xenium"
FILES = [
    ("processed_data.zip", 7.66e9),
    ("Y13-TUM-0-FO-2-S40-A1.zip", 2.14e9),
    ("BI29-TUM-0-FO-4-S4-C2.zip", 9.0e9),
    ("Y18-TUM-0-FO-1-S33-C1.zip", 10.07e9),
    ("BI24-TUM-0-FT-2-S29-B2.zip", 11.0e9),
    ("AQsb5-TUM-0-FT-1-S4-B1.zip", 11.92e9),
    ("Y22-TUM-0-FT-1-S4-B1.zip", 12.39e9),
    ("BI25-TUM-0-FT-2-S4-D1.zip", 20.11e9),
    ("Y14-TUM-0-FO-2-S30-C1.zip", 24.27e9),
    ("AQsb4-TUM-0-FO-1-S4-A1.zip", 31.97e9),
]


def human(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or u == "TB":
            return f"{n:.1f}{u}"
        n /= 1024


def download(fname, max_retries=100):
    url = f"https://zenodo.org/api/records/16785315/files/{fname}/content"
    os.makedirs(ROOT, exist_ok=True)
    dst = os.path.join(ROOT, fname)
    tmp = dst + ".part"
    if os.path.exists(dst):
        print(f"  [skip] {fname}")
        return
    t0 = time.time()
    for attempt in range(1, max_retries + 1):
        try:
            headers = {}
            if os.path.exists(tmp):
                headers["Range"] = f"bytes={os.path.getsize(tmp)}-"
            with requests.get(url, stream=True, timeout=(30, 300), headers=headers) as r:
                if r.status_code == 416:
                    os.rename(tmp, dst); print(f"\n  [done] {fname}"); return
                r.raise_for_status()
                mode = "ab" if "Range" in headers else "wb"
                got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(1024 * 256):
                        f.write(chunk); got += len(chunk)
                os.rename(tmp, dst)
                print(f"\n  [done] {fname} {human(os.path.getsize(dst))} in {time.time()-t0:.0f}s")
                return
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            print(f"\n  [retry {attempt}] {fname} at {human(got)}: {type(e).__name__}")
            time.sleep(5)
    print(f"\n  [FAILED] {fname}")


if __name__ == "__main__":
    for fname, _ in FILES:
        download(fname)
    print("=== XENIUM DOWNLOAD COMPLETE ===")
