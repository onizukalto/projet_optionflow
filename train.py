"""Construit les datasets, évalue (septembre = développement, août = holdout),
entraîne le modèle final et sauvegarde tout dans data/processed/ et models/.

Usage : python train.py   (aucune requête API si les données sont déjà sur le disque)
"""
from pathlib import Path

import joblib
import pandas as pd

from src.data import cached_days, download_day
from src.features import FLOW, PRICE, build_dataset
from src.model import ablation, make_model

PROCESSED = Path("data/processed")
PERIODS = {
    "dev_sept": ("2026-09-01", "2026-09-30"),
    "holdout_aug": ("2026-08-01", "2026-08-31"),
}
OPEX_DAYS = ["2026-08-21", "2026-09-18"]   # décidé avant de regarder août


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    datasets, results = {}, []
    for name, (start, end) in PERIODS.items():
        days = cached_days(start, end)
        print(f"{name} : {len(days)} jours")
        data = build_dataset(days, download_day)
        data.to_parquet(PROCESSED / f"{name}.parquet")
        datasets[name] = data

        for variant, subset in [("tous", data), ("hors_opex", data[~data["day"].isin(OPEX_DAYS)])]:
            res = ablation(subset, PRICE, FLOW).assign(period=name, variant=variant)
            results.append(res.reset_index())
            g = res["gain"]
            print(f"  {variant:10s} gain flow {g.mean():+.3f} ± {g.std() / len(g) ** .5:.3f} "
                  f"({(g > 0).sum()}/{len(g)} jours)")

    pd.concat(results).to_parquet(PROCESSED / "results.parquet")

    # modèle final : toutes les données, pour l'API et la démo
    full = pd.concat(datasets.values())
    model = make_model().fit(full[PRICE + FLOW], full["target"])
    joblib.dump({"model": model, "features": PRICE + FLOW}, "models/model.joblib")
    print("Modèle sauvegardé dans models/model.joblib")


if __name__ == "__main__":
    main()
