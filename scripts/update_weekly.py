from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from zerttracker.pipeline import run_weekly


def main() -> int:
    parser = argparse.ArgumentParser(description="Wöchentliches Update für DB Express-Zertifikate")
    parser.add_argument("--csv-fallback", type=Path, default=ROOT / "src" / "zerttracker" / "data" / "sample_certificates.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "cache" / "latest.parquet")
    parser.add_argument("--paths", type=int, default=20000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = run_weekly(csv_fallback=args.csv_fallback, output_path=args.output, n_paths=args.paths)
    if df.empty:
        print("Keine Daten erfasst.", file=sys.stderr)
        return 1
    print(df[["isin", "name", "exp_return_pa", "p_capital_loss", "score_total"]].head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
