"""Conservative secret scan before making the repository public.

Scans both the working tree's tracked files and every reasonably sized blob in
Git history. It reports file/object identifiers and rule names, never matching
secret contents.
"""

from __future__ import annotations

import json
import io
import re
import subprocess
from pathlib import Path


MAX_BLOB_BYTES = 5_000_000
PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(
        rb"(?:github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{20,})"
    ),
    "kaggle_token": re.compile(rb"KGAT_[A-Za-z0-9_-]{20,}"),
    "signed_url": re.compile(
        rb"(?:x-goog-signature|x-amz-signature|googleaccessid)=", re.IGNORECASE
    ),
    "literal_credential": re.compile(
        rb"(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
        rb"\s*[=:]\s*['\"][^'\"\s]{12,}['\"]",
        re.IGNORECASE,
    ),
}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def scan(data: bytes) -> list[str]:
    return [name for name, pattern in PATTERNS.items() if pattern.search(data)]


def main() -> None:
    tracked = [p for p in git("ls-files", "-z").decode().split("\0") if p]
    current_hits = []
    for name in tracked:
        path = Path(name)
        if not path.is_file() or path.stat().st_size > MAX_BLOB_BYTES:
            continue
        for rule in scan(path.read_bytes()):
            current_hits.append({"path": name, "rule": rule})

    object_paths = {}
    for row in git("rev-list", "--objects", "--all").splitlines():
        sha, *rest = row.split(b" ", 1)
        object_paths.setdefault(
            sha.decode(), rest[0].decode(errors="replace") if rest else ""
        )

    batch = subprocess.run(
        ["git", "cat-file", "--batch"],
        input="".join(f"{sha}\n" for sha in object_paths).encode(),
        capture_output=True,
        check=True,
    )
    stream = io.BytesIO(batch.stdout)

    history_hits = []
    scanned_blobs = 0
    while True:
        header = stream.readline()
        if not header:
            break
        sha_bytes, kind, size_bytes = header.rstrip(b"\n").split()
        size = int(size_bytes)
        data = stream.read(size)
        assert stream.read(1) == b"\n"
        if kind != b"blob" or size > MAX_BLOB_BYTES:
            continue
        scanned_blobs += 1
        sha = sha_bytes.decode()
        for rule in scan(data):
            history_hits.append(
                {"object": sha, "path": object_paths.get(sha, ""), "rule": rule}
            )

    report = {
        "tracked_files": len(tracked),
        "history_blobs_scanned": scanned_blobs,
        "max_blob_bytes": MAX_BLOB_BYTES,
        "current_hits": current_hits,
        "history_hits": history_hits,
        "passed": not current_hits and not history_hits,
        "note": "Matches identify locations and rule names only; secret values are never printed.",
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
