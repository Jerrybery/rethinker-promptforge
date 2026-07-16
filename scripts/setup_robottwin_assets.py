#!/usr/bin/env python
"""Create relative symlinks so the RoboTwin submodule can find assets.

RoboTwin expects asset trees under ``third_party/RoboTwin/assets/`` but those
assets are large and live in a separate checkout.  This script wires the
submodule to that checkout with portable relative symlinks that are recorded
in the main repo.

Example::

    python scripts/setup_robottwin_assets.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ASSET_NAMES = ["embodiments", "objects"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link RoboTwin asset trees into the submodule."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root of the rethinker-promptforge checkout.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=None,
        help="Directory containing the 'embodiments' and 'objects' asset trees. "
        "Defaults to '../rethinker-promptforge/assets' relative to the project root.",
    )
    return parser.parse_args(argv)


def _ensure_link(link: Path, target: Path) -> None:
    """Create or replace ``link`` as a relative symlink to ``target``."""
    if link.is_symlink():
        current = os.readlink(link)
        if Path(current) == target:
            print(f"OK: {link} -> {target}")
            return
        print(f"Replacing: {link} (was -> {current})")
        link.unlink()
    elif link.exists():
        raise FileExistsError(
            f"{link} already exists and is not a symlink; remove it manually."
        )

    link.symlink_to(target, target_is_directory=True)
    print(f"Linked: {link} -> {target}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = args.project_root.resolve()

    if args.assets_root is None:
        assets_root = (project_root.parent / "rethinker-promptforge" / "assets").resolve()
    else:
        assets_root = args.assets_root.resolve()

    link_dir = project_root / "third_party" / "RoboTwin" / "assets"
    link_dir.mkdir(parents=True, exist_ok=True)

    missing_targets: list[Path] = []
    for name in ASSET_NAMES:
        target = assets_root / name
        if not target.exists():
            missing_targets.append(target)
            continue

        link = link_dir / name
        rel_target = Path(os.path.relpath(target, link_dir))
        _ensure_link(link, rel_target)

    if missing_targets:
        print("Missing asset directories:", file=sys.stderr)
        for target in missing_targets:
            print(f"  - {target}", file=sys.stderr)
        print(
            "Provide them with --assets-root or place them at the default location.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
