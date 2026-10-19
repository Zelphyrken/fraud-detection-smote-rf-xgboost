"""End-to-end experiment aligned to the user's fraud-detection report.

Run from the project root:
    python run_experiment.py

The script writes EDA figures, metrics, confusion matrices, a selected model,
threshold metadata and a model-comparison CSV. Test data are not used for
hyperparameter tuning, threshold selection, or choosing the dashboard model.
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.common import (
    BASE_DIR,
    DATA_PATH,
    ENGINEERED_FEATURES,
    FIGURES_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    RAW_FEATURES,
    RESULTS_DIR,
    TARGET,
    engineer_features,
    find_threshold_max_f1,
    validate_dataset_columns,
)

# --------------------------- Experiment configuration -----------------------
DROP_EXACT_DUPLICATES = True
TEST_SIZE = 0.20          # held out until the final evaluation
VALIDATION_SIZE = 0.25    # 25% of the remaining 80% => 20% of total data
TUNING_MAX_ROWS = 40_000  # stratified subset from training ONLY to keep tuning practical
TUNING_ITERATIONS = 3
CV_FOLDS = 3
SEARCH_N_JOBS = 2          # modest parallelism helps avoid RAM pressure
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", category=FutureWarning)


def section(number: int, title: str) -> None:
    print("\n" + "=" * 88)
    print(f"LANGKAH {number}: {title}")
    print("=" * 88)


def save_class_distribution(y: pd.Series, path: Path, title: str) -> None:
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["Normal (Class 0)", "Fraud (Class 1)"], counts.values)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Jumlah transaksi")
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_eda_figures(data: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    save_class_distribution(
        data[TARGET], FIGURES_DIR / "class_distribution.png",
        "Distribusi kelas setelah audit data",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, name in [(0, "Normal"), (1, "Fraud")]:
        subset = data.loc[data[TARGET] == label, "Amount"]
        # The log x-scale makes the long-tailed amount distribution easier to read.
        subset = subset[subset >= 0]
        ax.hist(np.log1p(subset), bins=60, alpha=0.55, density=True, label=name)
    ax.set_title("Distribusi log(1 + Amount) menurut kelas")
    ax.set_xlabel("log(1 + Amount)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "amount_distribution.png", dpi=160)
    plt.close(fig)

    # Show the strongest absolute correlations with Class, without claiming causality.
    corr = data[RAW_FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    corr.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("15 fitur dengan korelasi absolut tertinggi terhadap Class")
    ax.set_xlabel("Korelasi Pearson")
    ax.set_ylabel("Fitur")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_correlations.png", dpi=160)
    plt.close(fig)


def make_pipeline(model, use_smote: bool) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if use_smote:
        # Scaling is fit only on each training fold; SMOTE is applied only inside
        # the training fold because both steps live inside imblearn Pipeline.
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)))
    steps.append(("classifier", model))
    return Pipeline(steps=steps)


def build_search(model_name: str, use_smote: bool):
    if model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        parameter_space = {
            "classifier__n_estimators": [100, 150, 250],
            "classifier__max_depth": [None, 8, 14],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", 0.5],
        }
    elif model_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
        )
        parameter_space = {
            "classifier__n_estimators": [100, 150, 250],
            "classifier__max_depth": [3, 4, 6],
            "classifier__learning_rate": [0.03, 0.1, 0.2],
            "classifier__subsample": [0.7, 0.9, 1.0],
            "classifier__colsample_bytree": [0.7, 0.9, 1.0],
            "classifier__min_child_weight": [1, 3, 5],
        }
    else:
        raise ValueError(f"Algoritma tidak dikenal: {model_name}")

    pipeline = make_pipeline(model, use_smote=use_smote)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_space,
        n_iter=TUNING_ITERATIONS,
        scoring="average_precision",
        cv=cv,
        refit=True,
        n_jobs=SEARCH_N_JOBS,
        random_state=RANDOM_STATE,
        verbose=1,
        error_score="raise",
        return_train_score=False,
    )
    return search


def calc_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        # Average Precision is used as the PR-AUC-style metric and is appropriate
        # for this highly imbalanced task. State the exact metric in the report.
        "PR-AUC (Average Precision)": average_precision_score(y_true, y_score),
        "ROC-AUC": roc_auc_score(y_true, y_score),
    }


def save_confusion_matrix(y_true, y_pred, title: str, filename: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Normal (0)", "Fraud (1)"],
    )
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close(fig)


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    for directory in (MODEL_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    section(1, "Pengumpulan dan pemeriksaan dataset")
    if not DATA_PATH.exists():
        print(f"Dataset belum ditemukan: {DATA_PATH}")
        print("Unduh creditcard.csv dari halaman Kaggle yang dicantumkan dalam README.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(column).strip() for column in df.columns]
    validate_dataset_columns(df)
    df = df.loc[:, RAW_FEATURES + [TARGET]].copy()
    print(f"Ukuran data awal: {df.shape[0]:,} baris x {df.shape[1]} kolom")
    print("Distribusi Class awal:")
    print(df[TARGET].value_counts(dropna=False).sort_index().to_string())

    section(2, "Data audit dan EDA")
    print("Missing values total:", int(df.isna().sum().sum()))
    print("Jumlah baris duplikat persis:", int(df.duplicated().sum()))
    print("Tipe data:")
    print(df.dtypes.to_string())
    if df.isna().any().any():
        # Dataset ULB should not have missing values, but fail loudly rather than
        # silently dropping potentially important minority-class rows.
        raise ValueError("Ditemukan missing value. Tinjau data sebelum melanjutkan eksperimen.")
    if not set(pd.unique(df[TARGET])).issubset({0, 1}):
        raise ValueError("Kolom Class harus berisi label 0 dan 1.")

    initial_rows = len(df)
    if DROP_EXACT_DUPLICATES:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Duplikat persis dihapus: {initial_rows - len(df):,}")
        print("Catatan: ukuran dan jumlah fraud akan berbeda dari angka dataset mentah di laporan.")
    else:
        print("Duplikat dipertahankan; pastikan tidak terjadi kebocoran duplikat train-test.")
    print(f"Ukuran data untuk pemodelan: {df.shape[0]:,} baris x {df.shape[1]} kolom")
    print("Distribusi Class untuk pemodelan:")
    print(df[TARGET].value_counts().sort_index().to_string())
    save_eda_figures(df)
    print(f"Grafik EDA disimpan di: {FIGURES_DIR}")

    section(3, "Preprocessing dan feature engineering")
    # Outliers are visualized/inspected in EDA but not automatically removed;
    # fraud transactions can be unusual by nature.
    y = df[TARGET].astype(int)
    X = engineer_features(df[RAW_FEATURES])
    print("Fitur model:", len(X.columns))
    print("Fitur tambahan: Amount_log1p = log(1 + Amount)")
    print("Outlier tidak dihapus otomatis.")

    section(4, "Stratified split: training, validation, testing")
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=VALIDATION_SIZE,
        stratify=y_train_all,
        random_state=RANDOM_STATE,
    )
    print(f"Train: {len(X_train):,}; fraud={int(y_train.sum()):,}")
    print(f"Validation: {len(X_val):,}; fraud={int(y_val.sum()):,}")
    print(f"Test (disimpan sampai evaluasi akhir): {len(X_test):,}; fraud={int(y_test.sum()):,}")

    tune_rows = min(TUNING_MAX_ROWS, len(X_train))
    if tune_rows < len(X_train):
        X_tune, _, y_tune, _ = train_test_split(
            X_train,
            y_train,
            train_size=tune_rows,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
        print(f"Tuning dilakukan pada subset stratified dari training: {len(X_tune):,} baris.")
    else:
        X_tune, y_tune = X_train, y_train
        print(f"Tuning menggunakan seluruh data training: {len(X_tune):,} baris.")

    section(5, "Training dan hyperparameter tuning Random Forest / XGBoost")
    all_models: dict[str, dict] = {}
    val_summary: list[dict] = []

    experiments = [
        ("Random Forest", False, "Random Forest | Tanpa SMOTE"),
        ("Random Forest", True, "Random Forest | SMOTE"),
        ("XGBoost", False, "XGBoost | Tanpa SMOTE"),
        ("XGBoost", True, "XGBoost | SMOTE"),
    ]

    for model_name, use_smote, experiment_name in experiments:
        print(f"\n--- {experiment_name} ---")
        search = build_search(model_name, use_smote)
        search.fit(X_tune, y_tune)
        print(f"Best CV Average Precision: {search.best_score_:.6f}")
        print("Best parameters:", search.best_params_)

        # Refit the selected hyperparameters on the full training partition.
        fitted_model = clone(search.best_estimator_)
        fitted_model.fit(X_train, y_train)
        validation_scores = fitted_model.predict_proba(X_val)[:, 1]
        threshold, validation_f1_at_threshold = find_threshold_max_f1(y_val, validation_scores)
        val_pr_auc = average_precision_score(y_val, validation_scores)
        val_roc_auc = roc_auc_score(y_val, validation_scores)
        val_prediction = (validation_scores >= threshold).astype(int)
        val_prec = precision_score(y_val, val_prediction, zero_division=0)
        val_rec = recall_score(y_val, val_prediction, zero_division=0)
        print(
            f"Validation: PR-AUC/AP={val_pr_auc:.6f}; ROC-AUC={val_roc_auc:.6f}; "
            f"threshold={threshold:.6f}; precision={val_prec:.4f}; "
            f"recall={val_rec:.4f}; F1={validation_f1_at_threshold:.4f}"
        )

        key = f"{model_name} | {'SMOTE' if use_smote else 'Tanpa SMOTE'}"
        all_models[key] = {
            "model": fitted_model,
            "threshold": threshold,
            "validation_pr_auc": float(val_pr_auc),
            "validation_roc_auc": float(val_roc_auc),
            "validation_f1": float(validation_f1_at_threshold),
            "best_cv_average_precision": float(search.best_score_),
            "best_params": search.best_params_,
            "algorithm": model_name,
            "sampling": "SMOTE" if use_smote else "Tanpa SMOTE",
        }
        val_summary.append({
            "Model": model_name,
            "Sampling": "SMOTE" if use_smote else "Tanpa SMOTE",
            "Validation Precision": val_prec,
            "Validation Recall": val_rec,
            "Validation F1-Score": validation_f1_at_threshold,
            "Validation PR-AUC (Average Precision)": val_pr_auc,
            "Validation ROC-AUC": val_roc_auc,
            "Threshold": threshold,
            "CV Best Average Precision": search.best_score_,
        })

    section(6, "Isolation Forest sebagai pembanding anomaly detection")
    # Unsupervised fit: labels are deliberately NOT passed into fit().
    iso_scaler = StandardScaler()
    X_train_scaled = iso_scaler.fit_transform(X_train)
    X_val_scaled = iso_scaler.transform(X_val)
    X_test_scaled = iso_scaler.transform(X_test)
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train_scaled)
    # Larger values of -decision_function indicate more anomalous observations.
    val_anomaly_scores = -iso.decision_function(X_val_scaled)
    iso_threshold, iso_validation_f1 = find_threshold_max_f1(y_val, val_anomaly_scores)
    iso_val_pred = (val_anomaly_scores >= iso_threshold).astype(int)
    iso_val_ap = average_precision_score(y_val, val_anomaly_scores)
    iso_val_roc = roc_auc_score(y_val, val_anomaly_scores)
    print(f"Validation PR-AUC/AP={iso_val_ap:.6f}; ROC-AUC={iso_val_roc:.6f}")
    print(f"Threshold anomaly={iso_threshold:.6f}; Validation F1={iso_validation_f1:.4f}")

    section(7, "Evaluasi akhir pada testing set")
    metric_rows: list[dict] = []
    test_predictions: dict[str, np.ndarray] = {}

    for key, info in all_models.items():
        estimator = info["model"]
        scores = estimator.predict_proba(X_test)[:, 1]
        predictions = (scores >= info["threshold"]).astype(int)
        metrics = calc_metrics(y_test, predictions, scores)
        display_name = key
        row = {
            "Model": info["algorithm"],
            "Sampling": info["sampling"],
            **metrics,
            "Threshold": info["threshold"],
            "Validation PR-AUC (Average Precision)": info["validation_pr_auc"],
        }
        metric_rows.append(row)
        test_predictions[display_name] = predictions
        save_confusion_matrix(
            y_test,
            predictions,
            f"Confusion Matrix - {display_name}\nthreshold={info['threshold']:.4f}",
            f"confusion_{safe_slug(display_name)}.png",
        )
        print(f"\n{display_name}")
        print(pd.Series(metrics).to_string(float_format=lambda value: f"{value:.6f}"))
        print("Confusion Matrix [[TN, FP], [FN, TP]]:")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))
        print("Classification report:")
        print(classification_report(y_test, predictions, target_names=["Normal", "Fraud"], zero_division=0, digits=4))

    # Isolated model comparator: its scores are anomaly scores rather than probabilities.
    iso_test_scores = -iso.decision_function(X_test_scaled)
    iso_test_pred = (iso_test_scores >= iso_threshold).astype(int)
    iso_metrics = calc_metrics(y_test, iso_test_pred, iso_test_scores)
    metric_rows.append({
        "Model": "Isolation Forest",
        "Sampling": "-",
        **iso_metrics,
        "Threshold": iso_threshold,
        "Validation PR-AUC (Average Precision)": iso_val_ap,
    })
    test_predictions["Isolation Forest"] = iso_test_pred
    save_confusion_matrix(
        y_test,
        iso_test_pred,
        f"Confusion Matrix - Isolation Forest\nthreshold={iso_threshold:.4f}",
        "confusion_isolation_forest.png",
    )
    print("\nIsolation Forest")
    print(pd.Series(iso_metrics).to_string(float_format=lambda value: f"{value:.6f}"))
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, iso_test_pred, labels=[0, 1]))

    results = pd.DataFrame(metric_rows)
    results.to_csv(RESULTS_DIR / "metrics_test.csv", index=False)
    pd.DataFrame(val_summary).to_csv(RESULTS_DIR / "metrics_validation.csv", index=False)

    # Choose the dashboard model using validation PR-AUC only, never test scores.
    best_key = max(all_models.keys(), key=lambda name: all_models[name]["validation_pr_auc"])
    best_info = all_models[best_key]
    joblib.dump(best_info["model"], MODEL_DIR / "best_model.joblib")
    joblib.dump(iso, MODEL_DIR / "isolation_forest.joblib")
    joblib.dump(iso_scaler, MODEL_DIR / "isolation_forest_scaler.joblib")

    metadata = {
        "selected_model": best_key,
        "selection_criterion": "Validation PR-AUC (Average Precision), not test performance",
        "threshold": float(best_info["threshold"]),
        "raw_feature_columns": RAW_FEATURES,
        "model_feature_columns": ENGINEERED_FEATURES,
        "threshold_selection": "Threshold maximizing F1-Score on validation data",
        "validation_pr_auc": float(best_info["validation_pr_auc"]),
        "validation_f1": float(best_info["validation_f1"]),
        "random_state": RANDOM_STATE,
        "note": "Prototype research output. Prediction is an indication, not proof of fraud.",
    }
    with open(RESULTS_DIR / "best_model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    # Save all four fitted supervised models and thresholds for future comparison.
    all_models_export = {
        key: {
            "model": info["model"],
            "threshold": info["threshold"],
            "algorithm": info["algorithm"],
            "sampling": info["sampling"],
            "validation_pr_auc": info["validation_pr_auc"],
            "best_params": info["best_params"],
        }
        for key, info in all_models.items()
    }
    joblib.dump(all_models_export, MODEL_DIR / "supervised_models.joblib", compress=3)

    section(8, "Selesai: file hasil eksperimen")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(results.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nTabel hasil testing : {RESULTS_DIR / 'metrics_test.csv'}")
    print(f"Tabel hasil validasi: {RESULTS_DIR / 'metrics_validation.csv'}")
    print(f"Metadata model      : {RESULTS_DIR / 'best_model_metadata.json'}")
    print(f"Model dashboard     : {MODEL_DIR / 'best_model.joblib'}")
    print(f"Grafik              : {FIGURES_DIR}")
    print(f"\nModel dashboard dipilih dari validation PR-AUC: {best_key}")
    print("Jangan mengubah hasil laporan menjadi angka lain. Gunakan angka yang benar-benar muncul dari run ini.")


if __name__ == "__main__":
    main()
