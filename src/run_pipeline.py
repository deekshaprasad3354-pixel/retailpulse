"""RetailPulse — end-to-end pipeline orchestrator.

Runs every stage in dependency order from one command:

    python -m src.run_pipeline                 # full run
    python -m src.run_pipeline --from churn    # resume from a stage
    python -m src.run_pipeline --only forecasting
    python -m src.run_pipeline --list

Stages are idempotent and cache their expensive inputs, so re-running after a
code change is cheap: only the stages you name actually recompute.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as cfg

# (key, human label, module path, function name)
STAGES: list[tuple[str, str, str, str]] = [
    ("download",   "Download raw UCI data",           "src.ingestion.download_data",   "main"),
    ("features",   "Ingest, clean, engineer features", "src.features.build_features",   "main"),
    ("segmentation", "Customer segmentation (RFM+KMeans)", "src.segmentation.segment_customers", "main"),
    ("forecasting", "Demand forecasting + backtest",   "src.forecasting.forecast_demand", "main"),
    ("churn",      "Churn prediction",                 "src.churn.predict_churn",       "main"),
    ("inventory",  "Inventory optimisation",           "src.inventory.optimize_inventory", "main"),
    ("drift",      "Drift detection",                  "src.mlops.drift",               "main"),
    ("tracking",   "MLflow experiment tracking",       "src.mlops.track_experiments",   "main"),
]


def stage_index(key: str) -> int:
    keys = [s[0] for s in STAGES]
    if key not in keys:
        raise SystemExit(f"unknown stage {key!r}. Valid stages: {', '.join(keys)}")
    return keys.index(key)


def run_stage(key: str, label: str, module: str, func: str, dry_run: bool = False) -> tuple[bool, float]:
    import importlib

    print("\n" + "=" * 78)
    print(f"STAGE [{key}] {label}")
    print("=" * 78)
    if dry_run:
        print("  (dry run — skipped)")
        return True, 0.0

    started = time.time()
    try:
        mod = importlib.import_module(module)
        getattr(mod, func)()
        elapsed = time.time() - started
        print(f"\n[OK] stage '{key}' finished in {elapsed:.1f}s")
        return True, elapsed
    except Exception:
        elapsed = time.time() - started
        print(f"\n[FAIL] stage '{key}' after {elapsed:.1f}s")
        traceback.print_exc()
        return False, elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the RetailPulse pipeline")
    ap.add_argument("--from", dest="from_stage", help="start at this stage")
    ap.add_argument("--only", help="run a single stage")
    ap.add_argument("--skip", nargs="*", default=[], help="stages to skip")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--dry-run", action="store_true", help="show the plan without running")
    args = ap.parse_args()

    if args.list:
        print("RetailPulse pipeline stages:\n")
        for i, (key, label, module, _) in enumerate(STAGES, 1):
            print(f"  {i}. {key:<14s} {label:<34s} ({module})")
        return

    if args.only:
        selected = [s for s in STAGES if s[0] == args.only]
        if not selected:
            raise SystemExit(f"unknown stage {args.only!r}")
    else:
        start = stage_index(args.from_stage) if args.from_stage else 0
        selected = STAGES[start:]

    selected = [s for s in selected if s[0] not in set(args.skip)]
    if not selected:
        raise SystemExit("nothing to run")

    print("=" * 78)
    print("RetailPulse — AI-Powered Customer Analytics & Demand Forecasting")
    print(f"Plan: {' -> '.join(s[0] for s in selected)}")
    print(f"Parquet cache: {'on' if cfg.USE_PARQUET else 'off (using gzip CSV)'}")
    print("=" * 78)

    started = time.time()
    results: list[tuple[str, bool, float]] = []
    for key, label, module, func in selected:
        ok, elapsed = run_stage(key, label, module, func, dry_run=args.dry_run)
        results.append((key, ok, elapsed))
        if not ok:
            print(f"\nStopping: stage '{key}' failed.")
            break

    total = time.time() - started
    print("\n" + "=" * 78)
    print("PIPELINE SUMMARY")
    print("=" * 78)
    for key, ok, elapsed in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {key:<14s} {elapsed:>8.1f}s")
    print(f"  {'-' * 32}")
    print(f"  total time: {total:.1f}s")
    failed = [k for k, ok, _ in results if not ok]
    print(f"\n  status: {'ALL STAGES PASSED' if not failed else 'FAILED: ' + ', '.join(failed)}")

    if not failed and not args.dry_run:
        print("\nNext steps:")
        print("  streamlit run dashboard/app.py     # launch the dashboard")
        print("  mlflow ui --backend-store-uri mlruns  # inspect experiments")


if __name__ == "__main__":
    main()
