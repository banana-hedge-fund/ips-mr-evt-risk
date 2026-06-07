# -*- coding: utf-8 -*-
"""ML-пайплайн: мета-лейблинг MR-сделок, сравнение методов, ML-фильтр.

Запуск: PYTHONPATH=. python scripts/run_ml.py
Результаты — в results/ (ml_*.csv, ml_*.png).
"""
import pathlib, warnings, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, roc_curve)

from src import data_io, ml

RES = pathlib.Path("results"); RES.mkdir(exist_ok=True)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]
MBY = {"2024": ["2024-03", "2024-08"], "2026": ["2026-01", "2026-04"]}


def main():
    frames = []
    for sym in SYMBOLS:
        for yr, ms in MBY.items():
            for m in ms:
                df = data_io.load_window(sym, [m])
                frames.append(ml.build_trade_dataset(df, sym, m, yr))
    D = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    print("Всего сделок:", len(D), "| доля прибыльных: %.3f" % D.label.mean())

    X = D[ml.FEATURES].values; y = D["label"].values
    tr = (D.year == "2024").values; te = (D.year == "2026").values
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    models = {
        "Логистическая L2": make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", C=1.0, max_iter=1000)),
        "Логистическая L1": make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="liblinear", C=0.5)),
        "kNN (k=25)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
        "Наивный Байес": GaussianNB(),
        "Решающее дерево": DecisionTreeClassifier(max_depth=4, random_state=0),
        "Случайный лес (бэггинг)": RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=-1, random_state=0),
        "Градиентный бустинг": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, C=1.0, gamma="scale", random_state=0)),
        "Нейросеть (MLP)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=500, early_stopping=True, random_state=0)),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    rows = []; proba_te = {}
    for name, mdl in models.items():
        try:
            cv = cross_val_score(mdl, Xtr, ytr, cv=tscv, scoring="roc_auc", n_jobs=-1).mean()
        except Exception:
            cv = np.nan
        mdl.fit(Xtr, ytr)
        p = mdl.predict_proba(Xte)[:, 1]
        proba_te[name] = p
        pred = (p >= 0.5).astype(int)
        rows.append({"model": name, "cv_auc": cv, "test_auc": roc_auc_score(yte, p),
                     "accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred, zero_division=0),
                     "recall": recall_score(yte, pred, zero_division=0), "f1": f1_score(yte, pred, zero_division=0)})
    R = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    R.to_csv(RES / "ml_model_comparison.csv", index=False)
    print(R.round(3).to_string(index=False))
    best = R.iloc[0]["model"]

    # ML-фильтр
    best_mdl = models[best]; ptr = best_mdl.predict_proba(Xtr)[:, 1]
    rn_tr = D.loc[tr, "ret_net"].values
    grid = np.quantile(ptr, np.linspace(0.1, 0.9, 33))
    thr = grid[int(np.argmax([rn_tr[ptr >= th].sum() for th in grid]))]
    pte = proba_te[best]; rn_te = D.loc[te, "ret_net"].values
    take = pte >= thr

    def stats(rn):
        eq = np.cumprod(1 + rn); dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
        cvar = -rn[rn <= np.quantile(rn, 0.05)].mean() if len(rn) else 0
        return dict(n=len(rn), total=float(np.prod(1 + rn) - 1), hit=float((rn > 0).mean()),
                    avg=float(rn.mean()), mdd=float(dd.min()), cvar5=float(cvar))
    cmp = pd.DataFrame([{"config": "baseline (все сделки)", **stats(rn_te)},
                        {"config": f"ML-фильтр (порог={thr:.2f})", **stats(rn_te[take])}])
    cmp.to_csv(RES / "ml_filter_compare.csv", index=False)
    print(cmp.round(4).to_string(index=False))

    # графики
    plt.figure(figsize=(7, 6))
    for name in R["model"]:
        fpr, tpr, _ = roc_curve(yte, proba_te[name])
        plt.plot(fpr, tpr, lw=1.3, label=f"{name} (AUC={roc_auc_score(yte, proba_te[name]):.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC (test=2026)"); plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig(RES / "ml_roc.png", dpi=120); plt.close()

    cm = confusion_matrix(yte, (pte >= 0.5).astype(int))
    plt.figure(figsize=(4.5, 4)); plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["убыточна", "прибыльна"]); plt.yticks([0, 1], ["убыточна", "прибыльна"])
    plt.xlabel("Прогноз"); plt.ylabel("Факт"); plt.title(f"Матрица ошибок: {best}")
    plt.tight_layout(); plt.savefig(RES / "ml_confusion.png", dpi=120); plt.close()

    rf = models["Случайный лес (бэггинг)"]; gb = models["Градиентный бустинг"]
    imp = pd.DataFrame({"feature": ml.FEATURES, "RF": rf.feature_importances_, "GB": gb.feature_importances_}).set_index("feature").sort_values("RF")
    imp.to_csv(RES / "ml_feature_importance.csv")
    imp.plot(kind="barh", figsize=(7, 5)); plt.title("Важность признаков")
    plt.tight_layout(); plt.savefig(RES / "ml_feature_importance.png", dpi=120); plt.close()

    order = np.argsort(D.loc[te, "exit_time"].values)
    eb = np.cumprod(1 + rn_te[order]); ef = np.cumprod(1 + np.where(take[order], rn_te[order], 0.0))
    plt.figure(figsize=(9, 4)); plt.plot(eb, label="baseline: все сделки"); plt.plot(ef, label="ML-фильтр")
    plt.title("Капитал по сделкам (test 2026)"); plt.xlabel("номер сделки"); plt.ylabel("equity")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(RES / "ml_equity_filter.png", dpi=120); plt.close()

    D.to_csv(RES / "ml_trades_dataset.csv", index=False)
    json.dump({"best": best, "threshold": float(thr)}, open(RES / "ml_summary.json", "w"), ensure_ascii=False, indent=2)
    print("Готово.")


if __name__ == "__main__":
    main()
