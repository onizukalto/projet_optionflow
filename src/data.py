"""Collecte des données Massive : prix QQQ et options 0DTE, avec cache disque."""
import os
import time
from pathlib import Path

import pandas as pd

RAW_ROOT = Path("data/raw")
SLEEP = 13            # secondes entre deux requêtes (plan gratuit ≈ 5 req/min)
STRIKE_RANGE = 0.01   # ±1 % autour du spot

_client = None


def get_client():
    """Crée le client Massive à la demande (la lecture du cache n'en a pas besoin)."""
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from massive import RESTClient
        load_dotenv()
        key = os.getenv("MASSIVE_API_KEY")
        if key is None:
            raise ValueError("MASSIVE_API_KEY not found in .env")
        _client = RESTClient(key)
    return _client


def api_call(method, *args, **kwargs):
    """Appelle client.<method>, attend SLEEP s, réessaie 3 fois en cas d'erreur."""
    fn = getattr(get_client(), method)
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
    """Relit le fichier s'il existe, sinon télécharge puis sauvegarde."""
    if path.exists():
        return pd.read_parquet(path)
    df = pd.DataFrame(fetch())
    df.to_parquet(path)
    return df


def to_ny(ts):
    return pd.to_datetime(ts, unit="ms", utc=True).dt.tz_convert("America/New_York")


def get_trading_days(start, end):
    days = pd.DataFrame(api_call("list_aggs", "QQQ", 1, "day", start, end))
    return to_ny(days["timestamp"]).dt.strftime("%Y-%m-%d").tolist()


def cached_days(start=None, end=None):
    """Journées déjà présentes sur le disque (aucune requête API)."""
    days = sorted(p.name for p in RAW_ROOT.iterdir() if (p / "_contracts.parquet").exists())
    return [d for d in days if (start is None or d >= start) and (end is None or d <= end)]


def download_day(day):
    """Renvoie (qqq, options) pour une journée ; télécharge seulement ce qui manque."""
    day_dir = RAW_ROOT / day
    day_dir.mkdir(parents=True, exist_ok=True)

    qqq = cached(day_dir / "_qqq.parquet",
                 lambda: api_call("list_aggs", "QQQ", 1, "minute", day, day, limit=50000))
    qqq["timestamp"] = to_ny(qqq["timestamp"])
    qqq = qqq.set_index("timestamp").between_time("09:30", "15:59").reset_index()
    spot = qqq["open"].iloc[0]
    lo, hi = spot * (1 - STRIKE_RANGE), spot * (1 + STRIKE_RANGE)

    contracts = cached(day_dir / "_contracts.parquet",
        lambda: api_call("list_options_contracts",
                         underlying_ticker="QQQ", expiration_date=day, as_of=day, expired=True,
                         strike_price_gte=lo, strike_price_lte=hi, limit=1000))
    contracts = contracts[contracts["strike_price"].between(lo, hi)]

    all_bars = []
    for _, row in contracts.iterrows():
        path = day_dir / f"{row['ticker'].replace(':', '_')}.parquet"
        bars = cached(path, lambda: api_call("list_aggs", row["ticker"], 1, "minute",
                                             day, day, limit=50000))
        if bars.empty:
            continue
        all_bars.append(bars.assign(ticker=row["ticker"],
                                    contract_type=row["contract_type"],
                                    strike_price=row["strike_price"]))

    if not all_bars:
        return qqq, None
    options = pd.concat(all_bars, ignore_index=True)
    options["timestamp"] = to_ny(options["timestamp"])
    return qqq, options
