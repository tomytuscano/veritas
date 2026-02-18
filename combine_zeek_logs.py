"""
combine_zeek_logs.py — Merge per-class Zeek CSVs into single files with a 'class' column.

For each CSV filename present in the class folders (bittorrent, normal, proxy, vpn),
reads every class's copy, appends a 'class' column set to the folder name, and writes
the combined result to zeek_logs/<filename>.

Usage:
    python combine_zeek_logs.py [--zeek-dir zeek_logs] [--output-dir zeek_logs]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

CLASSES = ["bittorrent", "normal", "proxy", "vpn"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def combine_by_filename(zeek_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect every unique CSV filename across all class folders.
    filenames: set[str] = set()
    for cls in CLASSES:
        folder = zeek_dir / cls
        if not folder.is_dir():
            log.warning("Folder not found, skipping: %s", folder)
            continue
        filenames.update(p.name for p in folder.glob("*.csv"))

    if not filenames:
        log.error("No CSV files found under %s", zeek_dir)
        return

    for filename in sorted(filenames):
        parts: list[pd.DataFrame] = []

        for cls in CLASSES:
            path = zeek_dir / cls / filename
            if not path.exists():
                log.warning("  [%s] missing %s — skipped", cls, filename)
                continue
            df = pd.read_csv(path, low_memory=False)
            df.insert(0, "class", cls)
            parts.append(df)
            log.info("  [%s] %s  →  %d rows", cls, filename, len(df))

        if not parts:
            log.warning("No data for %s, skipping output.", filename)
            continue

        combined = pd.concat(parts, ignore_index=True)
        out_path = output_dir / filename
        combined.to_csv(out_path, index=False)
        log.info("Wrote %d rows  →  %s", len(combined), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine per-class Zeek CSVs.")
    parser.add_argument(
        "--zeek-dir",
        default="zeek_logs",
        help="Root directory containing class sub-folders (default: zeek_logs)",
    )
    parser.add_argument(
        "--output-dir",
        default="zeek_logs",
        help="Directory to write combined CSVs (default: zeek_logs)",
    )
    args = parser.parse_args()

    zeek_dir = Path(args.zeek_dir)
    output_dir = Path(args.output_dir)

    log.info("Source : %s", zeek_dir.resolve())
    log.info("Output : %s", output_dir.resolve())

    combine_by_filename(zeek_dir, output_dir)


if __name__ == "__main__":
    main()
