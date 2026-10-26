import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from massive import RESTClient

load_dotenv()
client = RESTClient(os.environ["MASSIVE_API_KEY"])

RAW_ROOT = Path("data/raw")
SLEEP = 13
STRIKE_RANGE = 0.01


def api_call(fn, *args, **kwargs):
    for attempt in range(3):
        try:
            result = [vars(x) for x in fn(*args, **kwargs)]
            time.sleep(SLEEP)
            return result
        except Exception as e:
            print(f"  erreur {e.__class__.__name__}, pause 60 s ({attempt + 1}/3)", flush=True)
            time.sleep(60)
    raise RuntimeError("API indisponible après 3 tentatives")


def cached(path, fetch):
    if path.exists():
        return pd.read_parquet(path)
    df = pd.DataFrame(fetch())
    df.to_parquet(path)
    return df


def to_ny(ts):
    return pd.to_datetime(ts, unit="ms", utc=True).dt.tz_convert("America/New_York")


def download_day(day):
    day_dir = RAW_ROOT / day
    day_dir.mkdir(parents=True, exist_ok=True)

    qqq = cached(day_dir / "_qqq.parquet",
                 lambda: api_call(client.list_aggs, "QQQ", 1, "minute", day, day, limit=50000))
    qqq["timestamp"] = to_ny(qqq["timestamp"])
    qqq = qqq.set_index("timestamp").between_time("09:30", "15:59").reset_index()
    spot = qqq["open"].iloc[0]

    contracts = cached(day_dir / "_contracts.parquet",
        lambda: api_call(client.list_options_contracts,
                         underlying_ticker="QQQ", expiration_date=day, as_of=day, expired=True,
                         strike_price_gte=spot * (1 - STRIKE_RANGE),
                         strike_price_lte=spot * (1 + STRIKE_RANGE),
                         limit=1000))
    contracts = contracts[
        contracts["strike_price"].between(spot * (1 - STRIKE_RANGE), spot * (1 + STRIKE_RANGE))
    ]

    for i, (_, row) in enumerate(contracts.iterrows(), 1):
        path = day_dir / f"{row['ticker'].replace(':', '_')}.parquet"
        if path.exists():
            continue
        bars = cached(path, lambda: api_call(client.list_aggs, row["ticker"], 1, "minute",
                                             day, day, limit=50000))
        print(f"  [{i}/{len(contracts)}] {row['ticker']} : {len(bars)} barres", flush=True)


if __name__ == "__main__":
    start, end = sys.argv[1], sys.argv[2]
    days_df = pd.DataFrame(api_call(client.list_aggs, "QQQ", 1, "day", start, end))
    days = to_ny(days_df["timestamp"]).dt.strftime("%Y-%m-%d").tolist()
    print(f"{len(days)} jours de bourse : {days[0]} → {days[-1]}", flush=True)

    for day in days:
        print(f"=== {day} ===", flush=True)
        download_day(day)

    print("Terminé", flush=True)