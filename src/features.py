"""Transformation des données brutes d'une journée en table de features + target."""
import numpy as np
import pandas as pd

HORIZON = 5  # minutes

PRICE = ["return_1m", "return_5m", "return_15m"]
FLOW = [f"{name}_{w}m" for w in [5, 15, 30]
        for name in ["call_share_vol", "net_premium_share", "total_volume"]] + ["volume_vs_day_avg"]


def build_features(qqq, options, horizon=HORIZON):
    # flow par minute
    options = options.assign(premium=options["volume"] * options["vwap"] * 100)
    flow = options.pivot_table(index="timestamp", columns="contract_type",
                               values=["volume", "premium"], aggfunc="sum", fill_value=0)
    flow.columns = [f"{ctype}_{metric}" for metric, ctype in flow.columns]
    flow = flow.reindex(columns=["call_volume", "put_volume", "call_premium", "put_premium"],
                        fill_value=0)

    # une ligne par minute de séance
    df = qqq.set_index("timestamp")[["close"]].rename(columns={"close": "price"})
    df = df.join(flow, how="left").fillna(0)

    # features de flow relatives, sur plusieurs fenêtres (passé uniquement)
    for w in [5, 15, 30]:
        call_v = df["call_volume"].rolling(w).sum()
        put_v = df["put_volume"].rolling(w).sum()
        call_p = df["call_premium"].rolling(w).sum()
        put_p = df["put_premium"].rolling(w).sum()
        df[f"call_share_vol_{w}m"] = call_v / (call_v + put_v).replace(0, np.nan)
        df[f"net_premium_share_{w}m"] = (call_p - put_p) / (call_p + put_p).replace(0, np.nan)
        df[f"total_volume_{w}m"] = call_v + put_v
    df["volume_vs_day_avg"] = df["total_volume_5m"] / df["total_volume_5m"].expanding().mean()

    # features de prix (passé uniquement)
    df["return_1m"] = df["price"].pct_change(1)
    df["return_5m"] = df["price"].pct_change(5)
    df["return_15m"] = df["price"].pct_change(15)

    # target : seul regard vers le futur, jamais utilisé comme feature
    df["future_price"] = df["price"].shift(-horizon)
    df["target"] = (df["future_price"] > df["price"]).astype(int)

    return df.dropna()


def build_dataset(days, download_day):
    """Empile les journées ; `download_day` renvoie (qqq, options)."""
    frames = []
    for day in days:
        qqq, options = download_day(day)
        if options is None:
            continue
        frames.append(build_features(qqq, options).assign(day=day))
    return pd.concat(frames)
