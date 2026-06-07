#!/usr/bin/env python3
"""
One-shot converter for old Battledance Chess pickle snapshots.

Default usage, from the project folder:

    python convert_pkl_to_bdpop.py

That converts every ./models/**/*.pkl file to a sibling .bdpop file.
Use --delete-pkl after a successful conversion if you want the old pickle files
removed immediately.

Important: pickle is code-execution capable. Only run this on your own old local
training files, not on models received from someone else.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import battledance_training as bt  # type: ignore
except Exception as exc:  # pragma: no cover - user-facing startup failure
    raise SystemExit(f"Could not import battledance_training.py beside this script: {exc!r}")


def iter_pickle_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() == ".pkl":
            yield root
        return
    if not root.exists():
        return
    yield from sorted((p for p in root.rglob("*.pkl") if p.is_file()), key=lambda p: p.as_posix().lower())


def convert_one(path: Path, *, overwrite: bool, delete_pkl: bool, dry_run: bool) -> str:
    dst = path.with_suffix(".bdpop")
    if dst.exists() and not overwrite:
        return f"SKIP exists: {dst}"

    if dry_run:
        action = "replace" if dst.exists() else "write"
        return f"DRY  {path} -> {dst} ({action})"

    with path.open("rb") as f:
        payload = pickle.load(f)

    bt._safe_write_bdpop_payload(str(dst), payload, durable=True)  # pylint: disable=protected-access
    reread = bt._safe_read_bdpop_payload(str(dst))  # pylint: disable=protected-access
    if reread is None:
        raise RuntimeError(f"Converted file did not reload as BDPOP: {dst}")

    if delete_pkl:
        path.unlink()
        return f"OK   {path} -> {dst} ; deleted old .pkl"
    return f"OK   {path} -> {dst}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert old Battledance .pkl snapshots to pickle-free .bdpop files.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(SCRIPT_DIR / "models"),
        help="A .pkl file or a directory to scan recursively. Default: ./models",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing .bdpop output files.")
    parser.add_argument("--delete-pkl", action="store_true", help="Delete each old .pkl file after its .bdpop reloads successfully.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be converted without writing files.")
    args = parser.parse_args()

    root = Path(args.path).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()

    files = list(iter_pickle_files(root))
    if not files:
        print(f"No .pkl files found under {root}.")
        return 0

    ok = 0
    failed = 0
    for path in files:
        try:
            print(convert_one(path, overwrite=bool(args.overwrite), delete_pkl=bool(args.delete_pkl), dry_run=bool(args.dry_run)))
            ok += 1
        except Exception as exc:  # keep converting the rest
            failed += 1
            print(f"FAIL {path}: {exc!r}", file=sys.stderr)

    print(f"\nConverted/processed: {ok}; failed: {failed}.")
    if failed:
        print("Leave the old .pkl files in place until failures are resolved.", file=sys.stderr)
        return 1
    if not args.delete_pkl and not args.dry_run:
        print("Old .pkl files were left in place. Re-run with --delete-pkl to remove them after conversion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
