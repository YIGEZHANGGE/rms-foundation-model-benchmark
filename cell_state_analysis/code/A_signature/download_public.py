#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download public GEO supplementary files with resume + progress."""
import os
import sys
import time
import json
import requests

BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/{pre}/{acc}/suppl/"
ROOT = r"E:\rms_persister\data\public_sc"


def human(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or u == "TB":
            return f"{n:.1f}{u}"
        n /= 1024


def download(acc, fname, subdir=None, max_retries=50):
    pre = acc[:6] + "nnn"
    url = BASE.format(pre=pre, acc=acc) + fname
    d = os.path.join(ROOT, acc) if subdir is None else os.path.join(ROOT, acc, subdir)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, fname)
    if os.path.exists(dst):
        print(f"  [skip] {fname} exists")
        return
    tmp = dst + ".part"
    t0 = time.time()
    for attempt in range(1, max_retries + 1):
        try:
            headers = {}
            if os.path.exists(tmp):
                headers["Range"] = f"bytes={os.path.getsize(tmp)}-"
            with requests.get(url, stream=True, timeout=(30, 300), headers=headers) as r:
                if r.status_code == 416:
                    os.rename(tmp, dst)
                    print(f"\n  [done] {fname} (resumed complete)")
                    return
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                mode = "ab" if "Range" in headers else "wb"
                got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(1024 * 256):
                        f.write(chunk); got += len(chunk)
                # completed without exception
                os.rename(tmp, dst)
                dt = time.time() - t0
                print(f"\n  [done] {fname} {human(os.path.getsize(dst))} in {dt:.0f}s")
                return
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            print(f"\n  [retry {attempt}] {fname} broke at {human(got)}: {type(e).__name__}")
            time.sleep(3)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(f"\n  [skip 404] {fname}")
                return
            raise
    print(f"\n  [FAILED] {fname} after {max_retries} retries")


if __name__ == "__main__":
    # Danielli scRNA (already downloaded) + bulk combined TPM + Stewart RAW.tar
    tasks = [
        ("GSE218974", "GSE218974_RAW.tar"),
        ("GSE240308", "GSE240308_DYER-223923-STRANDED_RSEM_TPM.txt.gz"),
        ("GSE240287", "GSE240287_DYER-229970-STRANDED_RSEM_TPM.txt.gz"),
        ("GSE174376", "GSE174376_RAW.tar"),
    ]
    for acc, fn in tasks:
        download(acc, fn)
    print("=== ALL PUBLIC DATA DOWNLOADED ===")
