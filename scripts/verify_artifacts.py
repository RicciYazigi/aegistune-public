"""Verify an exported evidence manifest without model loading or network access."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent.resolve()
    files = manifest["files"]
    if not files:
        raise ValueError("Empty evidence manifest")
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Unsafe manifest path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact mismatch: {relative}")
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("ARTIFACTS.json"))
    args = parser.parse_args()
    count = verify(args.manifest)
    print(f"Verified {count} evidence/data/source-snapshot files. No inference performed.")


if __name__ == "__main__":
    main()
