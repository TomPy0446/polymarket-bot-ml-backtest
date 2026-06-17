"""Feature engineering for XGBoost ML layer.

Phase 1.2 — endogenous features only (parquet schemas don't include crypto
spot prices). Features are derived from resolved-market metadata +
sampled mid_price history.

Feature list (12 columns):
    volume                  : USD volume of the market at resolution time
    log_volume              : log1p(volume)
    mid_first               : first mid_price sample (window open)
    mid_last                : last mid_price sample (window close)
    mid_drift               : mid_last - mid_first
    mid_disp                : std of mid_price samples (within window)
    mid_n                   : number of mid_price samples (≤5)
    hour_sin, hour_cos      : sin/cos of hour-of-day UTC (period 24h)
    dow_sin,  dow_cos       : sin/cos of day-of-week UTC (period 7d)
    sym_BTC, sym_ETH, sym_SOL, sym_XRP, sym_DOGE, sym_OTHER : one-hot symbol

All missing values (markets without sampled mid_prices) are filled with
column medians computed on the training fold only — prevents leakage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dataset import (
    load_resolved, load_sampled, merge_resolved_with_sampled,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Frozen at train time — the schema is dumped alongside the model
    so live inference produces columns in the exact same order."""
    include_volume: bool = True
    include_mid_price: bool = True
    include_time_of_day: bool = True
    include_day_of_week: bool = True
    include_symbol: bool = True
    include_btc_exog: bool = False    # Phase 1b — off by default, opt-in
    medians: dict = field(default_factory=dict)   # filled at fit time

    def feature_columns(self) -> list[str]:
        cols = []
        if self.include_volume:
            cols += ["log_volume"]
        if self.include_mid_price:
            cols += ["mid_first", "mid_last", "mid_drift", "mid_disp", "mid_n"]
        if self.include_time_of_day:
            cols += ["hour_sin", "hour_cos"]
        if self.include_day_of_week:
            cols += ["dow_sin", "dow_cos"]
        if self.include_symbol:
            cols += ["sym_BTC", "sym_ETH", "sym_SOL", "sym_XRP", "sym_DOGE", "sym_OTHER"]
        if self.include_btc_exog:
            cols += [
                "btc_ret_5min", "btc_ret_15min", "btc_ret_60min",
                "btc_vol_15min", "btc_vol_60min",
                "btc_drift_15min", "btc_max_dd_60min",
                "btc_close_ratio_60min",   # close / rolling mean(60m) - 1
            ]
        return cols


TARGET_COL = "y_up"
SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "OTHER"]


def _add_time_features(df: pd.DataFrame, end_date_col: str = "end_date") -> pd.DataFrame:
    df = df.copy()
    end = pd.to_datetime(df[end_date_col], utc=True)
    hour = end.dt.hour + end.dt.minute / 60.0
    dow = end.dt.dayofweek  # 0=Mon, 6=Sun
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return df


def _add_symbol_features(df: pd.DataFrame, symbol_col: str = "symbol_canon") -> pd.DataFrame:
    df = df.copy()
    for s in SYMBOLS:
        df[f"sym_{s}"] = (df[symbol_col] == s).astype("int8")
    return df


def _add_volume_features(df: pd.DataFrame, volume_col: str = "volume") -> pd.DataFrame:
    df = df.copy()
    vol = pd.to_numeric(df[volume_col], errors="coerce").fillna(0.0)
    df["log_volume"] = np.log1p(vol.clip(lower=0.0))
    return df


def _add_btc_exog_features(
    df: pd.DataFrame,
    btc_klines: pd.DataFrame,
    end_date_col: str = "end_date",
) -> pd.DataFrame:
    """Add exogenous BTC features aligned to each market's end_date.

    Requires `btc_klines` indexed by UTC timestamp with 'close' column,
    1-minute resolution. Computes for each market:
        btc_ret_5min/15min/60min  : log(close_end / close_start)
        btc_vol_15min/60min       : std of 1m log-returns over window
        btc_drift_15min           : log(close_end / close_start)
        btc_max_dd_60min          : max drawdown over 60min window
        btc_close_ratio_60min     : close / mean(60m) - 1  (mean-reversion proxy)
    """
    df = df.copy()
    if btc_klines is None or btc_klines.empty or "close" not in btc_klines.columns:
        for col in (
            "btc_ret_5min", "btc_ret_15min", "btc_ret_60min",
            "btc_vol_15min", "btc_vol_60min",
            "btc_drift_15min", "btc_max_dd_60min", "btc_close_ratio_60min",
        ):
            df[col] = np.nan
        return df

    log_close = np.log(btc_klines["close"].astype(float))
    close = btc_klines["close"].astype(float)
    rets_1m = log_close.diff()

    end_dates = pd.to_datetime(df[end_date_col], utc=True)
    out = {col: np.full(len(df), np.nan, dtype=float) for col in (
        "btc_ret_5min", "btc_ret_15min", "btc_ret_60min",
        "btc_vol_15min", "btc_vol_60min",
        "btc_drift_15min", "btc_max_dd_60min", "btc_close_ratio_60min",
    )}

    # Pre-compute rolling means for btc_close_ratio_60min (mean of last 60min close)
    roll_mean_60 = close.rolling("60min").mean()

    for i, end_ts in enumerate(end_dates):
        if pd.isna(end_ts):
            continue
        end_ts = pd.Timestamp(end_ts)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")

        # 5-minute window
        win_5 = btc_klines[(btc_klines.index > end_ts - pd.Timedelta(minutes=5))
                          & (btc_klines.index <= end_ts)]
        if len(win_5) >= 2:
            out["btc_ret_5min"][i] = float(np.log(win_5["close"].iloc[-1] / win_5["close"].iloc[0]))

        # 15-minute window
        win_15 = btc_klines[(btc_klines.index > end_ts - pd.Timedelta(minutes=15))
                           & (btc_klines.index <= end_ts)]
        if len(win_15) >= 2:
            out["btc_ret_15min"][i] = float(np.log(win_15["close"].iloc[-1] / win_15["close"].iloc[0]))
            out["btc_vol_15min"][i] = float(rets_1m.loc[win_15.index].std()) if len(win_15) > 1 else 0.0
            out["btc_drift_15min"][i] = float(np.log(win_15["close"].iloc[-1] / win_15["close"].iloc[0]))

        # 60-minute window
        win_60 = btc_klines[(btc_klines.index > end_ts - pd.Timedelta(minutes=60))
                           & (btc_klines.index <= end_ts)]
        if len(win_60) >= 2:
            out["btc_ret_60min"][i] = float(np.log(win_60["close"].iloc[-1] / win_60["close"].iloc[0]))
            out["btc_vol_60min"][i] = float(rets_1m.loc[win_60.index].std()) if len(win_60) > 1 else 0.0
            cumax = win_60["close"].cummax()
            out["btc_max_dd_60min"][i] = float(((win_60["close"] / cumax) - 1).min())
            mean_60 = roll_mean_60.loc[end_ts] if end_ts in roll_mean_60.index else np.nan
            if not np.isnan(mean_60) and mean_60 > 0:
                out["btc_close_ratio_60min"][i] = float(win_60["close"].iloc[-1] / mean_60 - 1.0)

    for col, vals in out.items():
        df[col] = vals
    return df


def build_features(
    df: pd.DataFrame,
    cfg: FeatureConfig | None = None,
    fit_medians: bool = True,
    btc_klines: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Build the feature matrix from a resolved+merged DataFrame.

    Returns:
        X            : DataFrame of feature columns (no target)
        feature_cols : list of column names (frozen schema)
        medians      : dict {col: median} for filling NaNs at inference time

    If cfg.include_btc_exog is True, btc_klines (1m BTC OHLCV) must be supplied;
    otherwise the parameter is ignored.
    """
    if cfg is None:
        cfg = FeatureConfig()
    df = df.copy()
    df = _add_volume_features(df)
    df = _add_time_features(df)
    df = _add_symbol_features(df)
    if cfg.include_btc_exog:
        df = _add_btc_exog_features(df, btc_klines)

    feature_cols = cfg.feature_columns()
    X = df[feature_cols].copy()

    # Convert all to float
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Compute or apply medians for NaN-filling
    medians: dict = {}
    if fit_medians:
        for c in feature_cols:
            medians[c] = float(X[c].median()) if X[c].notna().any() else 0.0
        X = X.fillna(medians)
    else:
        for c in feature_cols:
            med = cfg.medians.get(c, 0.0)
            medians[c] = med
            X[c] = X[c].fillna(med)

    return X, feature_cols, medians


def build_inference_row(
    *,
    volume: float,
    mid_first: float | None,
    mid_last: float | None,
    mid_drift: float | None,
    mid_disp: float | None,
    mid_n: float | None,
    end_date: pd.Timestamp,
    symbol_canon: str,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame for live inference.

    Same column order as `build_features()`.
    """
    row = {
        "log_volume": float(np.log1p(max(0.0, volume or 0.0))),
        "mid_first": mid_first if mid_first is not None else np.nan,
        "mid_last":  mid_last  if mid_last  is not None else np.nan,
        "mid_drift": mid_drift if mid_drift is not None else np.nan,
        "mid_disp":  mid_disp  if mid_disp  is not None else np.nan,
        "mid_n":     mid_n     if mid_n     is not None else np.nan,
    }
    end = pd.to_datetime(end_date, utc=True)
    hour = end.hour + end.minute / 60.0
    dow = end.dayofweek
    row["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    row["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    row["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    row["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    for s in SYMBOLS:
        row[f"sym_{s}"] = int(symbol_canon == s)
    df = pd.DataFrame([row], columns=cfg.feature_columns())
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(cfg.medians.get(c, 0.0))
    return df


# ── Quick self-test when run directly ─────────────────────────────────────

def _selftest() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = load_resolved()
    s = load_sampled()
    merged = merge_resolved_with_sampled(r, s)
    cfg = FeatureConfig()
    X, cols, meds = build_features(merged, cfg)
    assert not X.isna().any().any(), "X should have no NaN after median fill"
    assert TARGET_COL in merged.columns
    y = merged[TARGET_COL].astype("int8")
    print(f"X.shape={X.shape}  cols={cols}")
    print(f"medians={meds}")
    print(f"y distribution: {y.value_counts().to_dict()}")
    print(f"feature importances baseline (corr with y):")
    for c in cols:
        corr = X[c].astype(float).corr(y.astype(float))
        print(f"  {c:12s}: {corr:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
