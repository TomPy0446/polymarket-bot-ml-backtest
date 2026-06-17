# polymarket-bot-ml-backtest — Implementation Plan

## Goal

Add three deliverables on top of `polymarket_bot/` (active v2 trading bot):

1. **XGBoost ML layer** — probability-of-Up model trained on resolved trades, blendable (50/50) with the existing Student-t heuristic signal. OFF by default, toggleable from dashboard.
2. **Walk-forward backtest framework** — extension of existing `backtest/engine.py` with ML-aware walk-forward harness, OOS Brier/log-loss/ECE metrics, regime breakdown.
3. **Streamlit dashboard exposed on `82.22.32.118`** via nginx vhost + TLS + basic auth (B2 binding strategy).

Source code lives in `/root/trading/polymarket_bot/` (existing active bot).
**No new capital is deployed**. The scan-only outsider bot in `/root/trading/outsider_scan/` provides ground-truth data for the ML training set.

## Anti-crash safeguards

- **Phase 0 mandatory** before any ML coding: validate parquet schemas + confirm `82.22.32.118` is public IP.
- **Atomic Git commits** after every validated step (conventional commits, Co-authored-by trailer).
- **Long-running scripts** (training, walk-forward backtest) launched via `nohup ... &` + log file, not blocking the session.
- **ML_ENABLED=false** by default — bot continues on existing heuristic until dashboard toggle.
- **Dashboard HTTPS** activated only in Phase 3, after ML + backtest validated.
- **Branch**: `feat/ml-xgb-backtest-dashboard` (isolated from main).

## Phase plan

```
PHASE 0 — Recon + Git setup    [~30 min]    → commit 1
├─ inspect_datasets.py (parquet schemas)
├─ curl ifconfig.io (confirm 82.22.32.118 public)
├─ branch feat/ml-xgb-backtest-dashboard
└─ infra/run_long.sh (nohup + log wrapper)

PHASE 1 — XGBoost ML layer      [~6-8h]    → commits 2-6
├─ 1.1: polymarket_bot/ml/inspect_datasets.py + dataset.py loaders   → commit 2
├─ 1.2: polymarket_bot/ml/features.py + unit tests                   → commit 3
├─ 1.3: polymarket_bot/ml/model.py (XGBProbabilisticModel)            → commit 4
├─ 1.4: polymarket_bot/ml/train.py CLI (CV + save to runs/<ts>/)     → commit 5
└─ 1.5: btc_updown.py + config.py (gated integration, OFF default)   → commit 6

PHASE 2 — Backtest extension    [~4-6h]    → commits 7-9
├─ 2.1: engine.py mods (gated expiry/stops) + metrics.py add        → commit 7
├─ 2.2: backtest/walk_forward.py + ml_evaluator.py                  → commit 8
├─ 2.3: backtest/report.py + cli/run_backtest_ml.py + classic.py   → commit 9
└─ smoke test: 1-fold async backtest                                → commit 9

PHASE 3 — Dashboard HTTPS       [~3-4h]    → commits 10-11
├─ 3.1: dashboard.py 3 new tabs (ML Model / Backtest Runs / Live Calibration) → commit 10
├─ 3.2: scripts/dashboard_https.sh + nginx vhost + self-signed cert  → commit 11
└─ smoke test: curl -k https://82.22.32.118/_stcore/health           → commit 11
```

## Decision locks

| Decision | Choice |
|---|---|
| XGBoost activation | **OFF by default**, dashboard toggle |
| Dashboard bind | **B2** (nginx + TLS + basic auth), Streamlit stays on 127.0.0.1:8501 |
| TLS cert | Self-signed (no DNS name available) |
| Data sources | Pre-built parquets `/root/backtest_data/{crypto15_resolved,sampled_prices}.parquet` + live enrichment later |
| Backtest reuse | Re-use existing `BacktestEngine`, `BacktestConfig`, `grid_search`, `metrics.py` |
| Walk-forward | 5 folds × 14d train / 3d test, embargo 30 min |

## Per-step "validated before commit" criteria

| Step | Validation gate |
|---|---|
| `inspect_datasets.py` | Prints schemas + n_rows + ts range |
| `features.py` | `build_features()` on 100 windows returns non-empty DataFrame |
| `model.py` | Fit on 1 fold → `predict_proba` shape correct |
| `train.py` | 1-fold async run completes without error, .joblib saved |
| `engine.py` mods | Existing `grid_search` regression test still passes |
| `walk_forward.py` | 1-fold async run writes `fold_metrics.json` |
| `dashboard.py` | Streamlit starts, 3 new tabs render on 127.0.0.1:8501 |
| `dashboard_https.sh` | `nginx -t` passes, `curl -k https://82.22.32.118/_stcore/health` → 200 |

## Cross-cutting risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Parquet schemas unknown | Gate everything on Phase 0 inspection |
| R2 | Book data absent in parquets → no OBI/depth | `FeatureConfig.depth_source = "book" \| "signed_volume" \| "skip"` |
| R3 | XGB training time | `n_jobs=-1`, `early_stopping_rounds=30`, fold-level parallelism option |
| R4 | Model overfit to window structure | Monotone constraints, min 200 OOS / fold, deflated Sharpe |
| R5 | Live drift | Weekly retrain, freshness-gated registry |
| R6 | tz-naive parquets | `--assume-utc` flag, assert in `load_*` |
| R7 | Missing `window_end_ts` | Re-use `btc_updown._detect_duration_from_market_name` |
| R8 | Pooled OOS hides bad fold | Always report worst-fold + regime breakdown |
| R9 | Self-signed cert browser warnings | Document exception, provide `dashboard_https.sh` regenerator |
| R10 | Basic auth weak | Sufficient for 1-2 trusted users; SSO is out of scope |
| R11 | Streamlit WS silent disconnect | `proxy_read_timeout 86400` + HTTP/1.1 upgrade in nginx |
| R12 | Port 443 firewalled | `ufw status` + check cloud security group |

## Session termination contract

Before any session ends:
- Either a full phase is committed ✅
- Or an async training/backtest is running with a logged PID, recoverable next session ✅
- **Never** leave uncommitted state.
