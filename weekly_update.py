#!/usr/bin/env python3
"""Refresh public EPL data and refit production only; never reselect on the final test."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


@contextmanager
def writer_lock(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Another update holds {path}. If it crashed, verify no writer is active before removing the lock.") from exc
    try:
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        yield
    finally:
        Path(path).unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", type=int, help="Start year; default is current season")
    p.add_argument("--skip-download", action="store_true", help="Refit from existing data without a network request")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    now = datetime.now(timezone.utc)
    season = args.season if args.season is not None else now.year if now.month >= 7 else now.year - 1
    models = root / "models"; models.mkdir(exist_ok=True)
    selection = root / "reports/selection.json"
    if not selection.is_file():
        p.error("Run train.py once to freeze walk-forward model selection first")
    run = root / "reports/updates" / now.strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True)
    with writer_lock(models / ".update.lock"):
        if not args.skip_download:
            subprocess.run([sys.executable, str(root / "data_ingest.py"), "--db", str(root / "data/epl.sqlite"),
                "--raw-dir", str(root / "data/raw"), "--interval", "3", "football-data", "--start-season", str(season - 1),
                "--end-season", str(season), "--refresh"], cwd=root, check=True)
        subprocess.run([sys.executable, str(root / "data_ingest.py"), "audit", "--output", str(run / "ingestion_audit.json")], cwd=root, check=True, stdout=subprocess.DEVNULL)
        from train import fit_production
        candidate = run / "production.joblib"
        bundle = fit_production(root / "data/epl.sqlite", selection, candidate)
        # Numeric checks happen before publication. This is not a profitability gate.
        for suffix in (".joblib", ".json"):
            source = candidate.with_suffix(suffix)
            temporary = models / ("production" + suffix + ".tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, models / ("production" + suffix))
        (run / "update.json").write_text(json.dumps({"run_at_utc": now.isoformat(), "season": season,
            "downloaded": not args.skip_download, "selection_unchanged": bundle["selection"]["classifier"],
            "snapshot_sha256": bundle["snapshot_sha256"], "information_through_utc": bundle["information_through_utc"]}, indent=2) + "\n")
    print(f"Updated production model. Version retained in {run}. Historical test reports are unchanged.")


if __name__ == "__main__":
    main()
