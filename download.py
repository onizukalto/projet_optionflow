"""Téléchargement des données brutes.  Usage : python download.py 2026-08-01 2026-08-31"""
import sys

from src.data import download_day, get_trading_days

if __name__ == "__main__":
    start, end = sys.argv[1], sys.argv[2]
    days = get_trading_days(start, end)
    print(f"{len(days)} jours de bourse : {days[0]} → {days[-1]}", flush=True)
    for day in days:
        print(f"=== {day} ===", flush=True)
        download_day(day)
    print("Terminé", flush=True)
