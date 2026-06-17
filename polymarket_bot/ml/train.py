"""Train XGBoost ML model via walk-forward CV.

Phase 1.4 — CLI entrypoint that:
  1. Loads crypto15_resolved.parquet + sampled_prices.parquet
  2. Generates K time-based folds with embargo
  3. For each fold: fit XGB on train, calibrate on a 15% tail of train,
     predict OOS on test, compute Brier/log-loss/accuracy/AUC
  4. Aggregates pooled OOS predictions across folds
  5. Refits final model on ALL data with same hyperparams
  6. Saves everything to runs/<UTC-timestamp>/ml/

Usage:
    python -m polymarket_bot.ml.train \\
        --folds 3 --train-days 14 --test-days 3 --embargo-min 30 \\
        --out runs/20260617T200000Z/ml/

    # Async (long training):
    bash infra/run_long.sh ml-train \\
        "python -m polymarket_bot.ml.train --folds 5 --train-days 21"
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)

from .dataset import (
    load_resolved, load_sampled, merge_resolved_with_sampled,
    walk_forward_splits, split_mask,
)
from .features import build_features, FeatureConfig, TARGET_COL
from .model import XGBConfig, XGBProbabilisticModel, ModelMetadata

logger = logging.getLogger(__name__)


def compute_ece(y_true, p_up, n_bins: int = 10) -> float:
    """Expected Calibration Error — binned reliability gap."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p_up, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        avg_p = p_up[mask].mean()
        avg_y = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(avg_p - avg_y)
    return float(ece)


def evaluate_fold(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    out = {
        "n": int(len(y_test)),
        "brier": float(brier_score_loss(y_test, proba)),
        "log_loss": float(log_loss(y_test, np.clip(proba, 1e-15, 1 - 1e-15))),
        "accuracy": float(accuracy_score(y_test, preds)),
        "ece": float(compute_ece(y_test, proba)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_test, proba))
    except ValueError:
        out["auc"] = float("nan")
    return out


def train_walk_forward(
    *,
    n_folds: int = 3,
    train_days: int = 14,
    test_days: int = 3,
    embargo_minutes: int = 30,
    xgb_cfg: XGBConfig | None = None,
    out_dir: str,
    include_btc_exog: bool = False,
    btc_symbol: str = "BTCUSDT",
    btc_interval: str = "1m",
) -> dict:
    """Run walk-forward CV and save artefacts to out_dir.
    Returns a summary dict with per-fold metrics, pooled OOS metrics,
    and the path to the final saved model.
    """
    if xgb_cfg is None:
        xgb_cfg = XGBConfig()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ML TRAIN — walk-forward CV")
    logger.info("=" * 70)
    logger.info("  folds=%d  train_days=%d  test_days=%d  embargo_min=%d  "
                "include_btc_exog=%s",
                n_folds, train_days, test_days, embargo_minutes, include_btc_exog)

    # 1. Load + merge
    resolved = load_resolved()
    sampled = load_sampled()
    merged = merge_resolved_with_sampled(resolved, sampled)
    logger.info("  dataset: %d markets after merge", len(merged))

    # 1b. Fetch BTC klines (optional) — bounded to the union of fold windows
    # (not the full dataset span — that would be 11+ months of 1m candles
    # which is unnecessary since folds only cover the last few weeks).
    btc_klines: pd.DataFrame | None = None
    if include_btc_exog:
        from .binance_klines import fetch_klines_cached
        # The folds are anchored on the end of the dataset (last test window
        # = most recent n_folds × test_days). So we only need BTC klines for
        # the union of all fold spans, plus buffer for the largest window.
        fold_total_days = (train_days + test_days) * n_folds + 2  # +2 buffer
        span_end_dt = merged["end_date"].max() + pd.Timedelta(hours=1)
        span_start_dt = span_end_dt - pd.Timedelta(days=fold_total_days)
        span_start = span_start_dt.timestamp()
        span_end = span_end_dt.timestamp()
        logger.info("  fetching BTC klines %s [%s .. %s] (last %d days for %d folds) ...",
                    btc_symbol,
                    pd.Timestamp(span_start, unit="s", tz="UTC"),
                    pd.Timestamp(span_end, unit="s", tz="UTC"),
                    fold_total_days, n_folds)
        btc_klines = fetch_klines_cached(
            symbol=btc_symbol, interval=btc_interval,
            start_ts=span_start, end_ts=span_end,
        )
        logger.info("  btc_klines: %d rows [%s .. %s]",
                    len(btc_klines), btc_klines.index.min(), btc_klines.index.max())

    # 2. Walk-forward splits
    splits = walk_forward_splits(
        merged, n_folds=n_folds, train_days=train_days,
        test_days=test_days, embargo_minutes=embargo_minutes,
    )
    if not splits:
        raise RuntimeError("walk_forward produced 0 folds — check train_days / dataset span")

    # 3. Per-fold train + evaluate
    fold_metrics: list[dict] = []
    pooled_p: list[np.ndarray] = []
    pooled_y: list[np.ndarray] = []
    fold_models: list[XGBProbabilisticModel] = []

    cfg = FeatureConfig(include_btc_exog=include_btc_exog)
    feature_cols = cfg.feature_columns()
    y_all = merged[TARGET_COL].astype("int8").values

    for split in splits:
        t0 = time.time()
        train_m, test_m = split_mask(merged, split)
        train_df = merged.loc[train_m].copy()
        test_df = merged.loc[test_m].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            logger.warning("fold %d empty — skipping", split.fold_id)
            continue

        # Build features (medians computed on train fold only)
        X_train, _, medians = build_features(train_df, cfg, fit_medians=True, btc_klines=btc_klines)
        X_test, _, _ = build_features(test_df, cfg, fit_medians=False, btc_klines=btc_klines)
        X_test = X_test.fillna(medians)  # belt-and-suspenders
        y_train = train_df[TARGET_COL].astype("int8").values
        y_test = test_df[TARGET_COL].astype("int8").values

        # Hold out last 15% of train for calibrator
        n = len(X_train)
        cal_size = max(int(n * 0.15), 200)
        X_fit, X_cal = X_train.iloc[:n - cal_size], X_train.iloc[n - cal_size:]
        y_fit, y_cal = y_train[:n - cal_size], y_train[n - cal_size:]

        model = XGBProbabilisticModel(
            cfg=xgb_cfg, feature_columns=feature_cols,
            feature_medians=medians, use_isotonic=True,
        )
        model.fit(X_fit, y_fit, eval_set=[(X_cal, y_cal)], verbose=False)
        model.calibrate(X_cal, y_cal)

        metrics = evaluate_fold(model, X_test, y_test)
        elapsed = time.time() - t0
        metrics.update({
            "fold_id": split.fold_id,
            "train_period": [str(split.train_start), str(split.train_end)],
            "test_period": [str(split.test_start), str(split.test_end)],
            "n_train": int(len(y_fit)),
            "n_cal": int(len(y_cal)),
            "n_test": int(len(y_test)),
            "elapsed_s": round(elapsed, 1),
        })
        fold_metrics.append(metrics)
        fold_models.append(model)

        # Pool OOS predictions
        proba_oos = model.predict_proba(X_test)[:, 1]
        pooled_p.append(proba_oos)
        pooled_y.append(y_test)

        logger.info("  fold %d: n_train=%d n_test=%d  brier=%.4f  acc=%.3f  "
                    "elapsed=%.1fs",
                    split.fold_id, len(y_fit), len(y_test),
                    metrics["brier"], metrics["accuracy"], elapsed)

    if not pooled_p:
        raise RuntimeError("no folds produced predictions")

    pooled_p_arr = np.concatenate(pooled_p)
    pooled_y_arr = np.concatenate(pooled_y)
    pooled_metrics = {
        "n_oos_total": int(len(pooled_y_arr)),
        "brier": float(brier_score_loss(pooled_y_arr, pooled_p_arr)),
        "log_loss": float(log_loss(pooled_y_arr, np.clip(pooled_p_arr, 1e-15, 1 - 1e-15))),
        "accuracy": float(accuracy_score(pooled_y_arr, (pooled_p_arr >= 0.5).astype(int))),
        "ece": float(compute_ece(pooled_y_arr, pooled_p_arr)),
    }
    try:
        pooled_metrics["auc"] = float(roc_auc_score(pooled_y_arr, pooled_p_arr))
    except ValueError:
        pooled_metrics["auc"] = float("nan")
    logger.info("  POOLED OOS: n=%d  brier=%.4f  log_loss=%.4f  acc=%.3f  ece=%.4f",
                pooled_metrics["n_oos_total"], pooled_metrics["brier"],
                pooled_metrics["log_loss"], pooled_metrics["accuracy"], pooled_metrics["ece"])

    # 4. Final refit on ALL data, calibrator on 15% tail
    logger.info("Final refit on full dataset (%d rows)...", len(merged))
    X_all, _, medians_full = build_features(merged, cfg, fit_medians=True, btc_klines=btc_klines)
    y_all = merged[TARGET_COL].astype("int8").values
    n = len(X_all)
    cal_size = max(int(n * 0.15), 200)
    X_fit, X_cal = X_all.iloc[:n - cal_size], X_all.iloc[n - cal_size:]
    y_fit, y_cal = y_all[:n - cal_size], y_all[n - cal_size:]
    final_model = XGBProbabilisticModel(
        cfg=xgb_cfg, feature_columns=feature_cols,
        feature_medians=medians_full, use_isotonic=True,
    )
    final_model.fit(X_fit, y_fit, eval_set=[(X_cal, y_cal)], verbose=False)
    final_model.calibrate(X_cal, y_cal)
    final_model.metadata_ = ModelMetadata(
        version=f"xgb_v{int(time.time())}",
        trained_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        n_train=int(len(y_fit)),
        n_oos=int(pooled_metrics["n_oos_total"]),
        oos_brier=pooled_metrics["brier"],
        oos_log_loss=pooled_metrics["log_loss"],
        oos_accuracy=pooled_metrics["accuracy"],
        feature_columns=feature_cols,
        feature_medians=medians_full,
        xgb_config=asdict(xgb_cfg),
        isotonic_calibrated=True,
        notes=f"walk-forward CV: {len(fold_metrics)} folds × {train_days}d train / {test_days}d test, embargo {embargo_minutes}min",
    )
    final_model.save(str(out_path / "model"))
    logger.info("  final model saved to %s/model", out_path)

    # 5. Save summary
    summary = {
        "config": {
            "n_folds": n_folds, "train_days": train_days,
            "test_days": test_days, "embargo_minutes": embargo_minutes,
            "xgb_cfg": asdict(xgb_cfg),
        },
        "folds": fold_metrics,
        "pooled_oos": pooled_metrics,
        "model_path": str(out_path / "model"),
        "feature_columns": feature_cols,
        "feature_importances": {
            c: float(imp)
            for c, imp in zip(feature_cols, final_model.model_.feature_importances_)
        },
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("  summary saved to %s/summary.json", out_path)
    return summary


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Train XGBoost model via walk-forward CV")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--train-days", type=int, default=14)
    p.add_argument("--test-days", type=int, default=3)
    p.add_argument("--embargo-min", type=int, default=30)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--include-btc-exog", action="store_true",
                   help="Add exogenous BTC features (returns, vol, drawdown). Requires Binance fetch.")
    p.add_argument("--btc-symbol", type=str, default="BTCUSDT")
    p.add_argument("--btc-interval", type=str, default="1m")
    p.add_argument("--out", type=str, default=None,
                   help="Output dir (default: runs/<UTC>/ml/)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.out is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        args.out = f"/root/trading/runs/{ts}/ml"

    cfg = XGBConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    summary = train_walk_forward(
        n_folds=args.folds,
        train_days=args.train_days,
        test_days=args.test_days,
        embargo_minutes=args.embargo_min,
        xgb_cfg=cfg,
        include_btc_exog=args.include_btc_exog,
        btc_symbol=args.btc_symbol,
        btc_interval=args.btc_interval,
        out_dir=args.out,
    )
    # Brief summary to stdout
    pooled = summary["pooled_oos"]
    print()
    print("=" * 60)
    print(f"POOLED OOS  brier={pooled['brier']:.4f}  log_loss={pooled['log_loss']:.4f}  "
          f"acc={pooled['accuracy']:.3f}  ece={pooled['ece']:.4f}  auc={pooled['auc']:.3f}")
    print(f"Top features:")
    fi = sorted(summary["feature_importances"].items(), key=lambda x: -x[1])[:5]
    for name, imp in fi:
        print(f"  {name:12s}: {imp:.4f}")
    print(f"Model: {summary['model_path']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
