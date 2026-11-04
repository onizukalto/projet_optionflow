# QQQ Options Flow → 5-minute Direction

**Does short-term QQQ 0DTE options flow contain predictive information about the next 5-minute move of the Nasdaq-100?**

End-to-end project: market-data ingestion, options-flow feature engineering, leakage-free
evaluation, an inference API and an interactive dashboard.

## Result

**No exploitable signal.** On the development month (September 2026) adding options-flow features
to price features seemed to help slightly. On a held-out month (August 2026), tested once with a
frozen method, the effect disappeared.

| AUC (walk-forward, mean) | Price only | Price + flow | Flow contribution | Days flow helps |
|---|---|---|---|---|
| September — development | 0.491 | 0.519 | +0.027 ± 0.017 | 10 / 13 |
| **August — holdout** | **0.516** | **0.502** | **−0.015 ± 0.010** | **7 / 16** |
| August — excl. monthly expiry | 0.519 | 0.503 | −0.015 ± 0.011 | 6 / 15 |

The September gain was most likely noise amplified by feature choices made while looking at that
month. The holdout is what caught it.

![Dashboard](docs/dashboard.png)

## Pipeline

```text
Massive API ──► data/raw/<day>/*.parquet      (cached, rate-limited)
                    │
                    ▼
            build_features()                    one row per minute
                    │
                    ▼
     walk-forward Random Forest               price-only vs price + flow
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    FastAPI /predict    Streamlit dashboard
```

```text
src/data.py       Massive client, disk cache, retry on HTTP 429
src/features.py   options-flow and price features, 5-minute target
src/model.py      Random Forest, walk-forward evaluation, ablation
download.py       download raw data for a date range
train.py          build datasets, evaluate, save results and final model
api/main.py       inference API
app.py            Streamlit dashboard
tests/            feature, leakage and API tests
train.ipynb       research notebook
```

## Method

- **Data** — 1-minute bars of QQQ and of 0DTE QQQ options with strikes within ±1 % of the
  opening price, regular session only (09:30–16:00 New York).
- **Features** (past data only) — share of volume on calls, net premium share (−1 = all puts,
  +1 = all calls) and total volume over 5 / 15 / 30 minutes; recent volume vs. day average;
  1 / 5 / 15-minute returns.
- **Target** — 1 if the price is higher 5 minutes later, computed within each day.
- **Evaluation** — walk-forward: each day is predicted by a model trained only on earlier days.
  Ablation of price-only vs. price + flow, compared day by day with ROC AUC.
- **Protocol** — September for development; August as a holdout evaluated once with the method
  frozen. The monthly-expiry exclusion was specified before looking at August.
- **Leakage guard** — `tests/test_features.py` checks that altering future prices and volumes
  leaves every past feature unchanged.

## Run it

```bash
pip install -r requirements.txt
echo "MASSIVE_API_KEY=your_key" > .env

python download.py 2026-08-01 2026-09-30   # ~7 min per day on the free plan
python train.py                            # evaluation + models/model.joblib
streamlit run app.py                       # dashboard on :8501
uvicorn api.main:app --reload              # API on :8000, docs at /docs
pytest                                     # tests
```

With Docker, after `python train.py`:

```bash
docker compose up --build    # API on :8000, dashboard on :8501
```

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "return_1m": 0.0002, "return_5m": -0.001, "return_15m": 0.0015,
  "call_share_vol_5m": 0.62, "net_premium_share_5m": 0.3, "total_volume_5m": 9000,
  "call_share_vol_15m": 0.58, "net_premium_share_15m": 0.2, "total_volume_15m": 26000,
  "call_share_vol_30m": 0.55, "net_premium_share_30m": 0.1, "total_volume_30m": 51000,
  "volume_vs_day_avg": 1.3}'
```

## Limitations

- Two months of data (~35 test days): small effects cannot be ruled out.
- 5-minute horizon; transaction costs and execution are not modelled.
- QQQ is used as the target, as a proxy for Nasdaq-100 futures (NQ).
- Flow is measured from aggregated minute bars: trade direction (buyer- or seller-initiated)
  is unknown, so call volume mixes bullish buying and call selling.
- Only 0DTE contracts near the money; no open interest, Greeks or dealer-positioning (GEX) features.

## Next steps

- Replace QQQ with NQ minute bars as the target.
- Classify trades as buy/sell from quotes to get a signed flow.
- Track experiments with MLflow.
