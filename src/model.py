"""Modèle et évaluation par validation glissante (walk-forward)."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

MIN_TRAIN_DAYS = 5


def make_model():
    return RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=20,
                                  random_state=42, n_jobs=-1)


def walk_forward(dataset, features, min_train_days=MIN_TRAIN_DAYS):
    """Pour chaque jour : entraîne sur les jours précédents, renvoie l'AUC du jour."""
    days = sorted(dataset["day"].unique())
    aucs = {}
    for i in range(min_train_days, len(days)):
        train = dataset[dataset["day"].isin(days[:i])]
        test = dataset[dataset["day"] == days[i]]
        model = make_model().fit(train[features], train["target"])
        aucs[days[i]] = roc_auc_score(test["target"], model.predict_proba(test[features])[:, 1])
    return pd.Series(aucs, name="auc")


def ablation(dataset, price, flow):
    """Compare prix seul vs prix + flow, jour par jour."""
    res = pd.DataFrame({
        "prix_seul": walk_forward(dataset, price),
        "prix_flow": walk_forward(dataset, price + flow),
    })
    res["gain"] = res["prix_flow"] - res["prix_seul"]
    res.index.name = "day"
    return res
