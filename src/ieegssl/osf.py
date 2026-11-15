"""Minimal OSF API v2 client (standard library only).

Lists every file in an OSF project's osfstorage (recursing into folders) and
downloads it with resume support and checksum verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.osf.io/v2"
USER_AGENT = "ieegssl-downloader/1.0"


def _get_json(url: str, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # transient network error: retry with backoff
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  listing failed ({exc}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def parse_listing(payload: dict) -> tuple[list[dict], list[str], str | None]:
    """Return (files, folder_urls, next_page_url) from one OSF listing page."""
    files, folders = [], []
    for item in payload.get("data", []):
        attrs = item.get("attributes", {})
        kind = attrs.get("kind")
        if kind == "file":
            hashes = (attrs.get("extra") or {}).get("hashes") or {}
            files.append(
                {
                    "name": attrs.get("name"),
                    "path": attrs.get("materialized_path") or attrs.get("name"),
                    "size": attrs.get("size"),
                    "md5": hashes.get("md5"),
                    "sha256": hashes.get("sha256"),
                    "download": (item.get("links") or {}).get("download"),
                }
            )
        elif kind == "folder":
            rel = (item.get("relationships") or {}).get("files") or {}
            href = ((rel.get("links") or {}).get("related") or {}).get("href")
            if href:
                folders.append(href)
    next_url = (payload.get("links") or {}).get("next")
    return files, folders, next_url


def list_files(node: str) -> list[dict]:
    start = f"{API}/nodes/{node}/files/osfstorage/?" + urllib.parse.urlencode({"page[size]": 100})
    queue, files = [start], []
    while queue:
        url = queue.pop(0)
        while url:
            page_files, folders, url = parse_listing(_get_json(url))
            files.extend(page_files)
            queue.extend(folders)
    return files


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, meta: dict) -> bool:
    if not path.exists():
        return False
    if meta.get("size") is not None and path.stat().st_size != int(meta["size"]):
        return False
    if meta.get("sha256"):
        return _digest(path, "sha256") == meta["sha256"]
    if meta.get("md5"):
        return _digest(path, "md5") == meta["md5"]
    return True


def download(meta: dict, dest_dir: Path, retries: int = 5) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / meta["name"]
    if verify(target, meta):
        return target
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(retries):
        try:
            done = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": USER_AGENT}
            if done:
                headers["Range"] = f"bytes={done}-"
            req = urllib.request.Request(meta["download"], headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                mode = "ab" if done and resp.status == 206 else "wb"
                with open(part, mode) as fh:
                    while True:
                        chunk = resp.read(1 << 22)
                        if not chunk:
                            break
                        fh.write(chunk)
            os.replace(part, target)
            if verify(target, meta):
                return target
            print(f"  checksum mismatch for {meta['name']}; re-downloading", flush=True)
            target.unlink(missing_ok=True)
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  {meta['name']}: {exc}; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {meta['name']}")
