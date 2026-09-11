"""
RQ2: MLP — Feature Augmentation
================================
Research Question: Does adding ACE and/or SDHE features improve prediction
of frequent mental distress beyond traditional features?

4 Feature Configurations:
  (A) Baseline:              Traditional features only (same as RQ1)
  (B) Baseline + ACE:        + 13 ACE items + ACE_score
  (C) Baseline + SDHE:       + 10 SDHE items + SDHE_burden
  (D) Baseline + ACE + SDHE + interaction: All features

Usage:
  python RQ2_MLP.py
"""

import ast
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH = "df_rq2_augmentation.csv"
SAVE_DIR  = os.path.expanduser("~/Desktop/ML A2 data")


# ---------------------------------------------------------------------------
# Feature configurations
# ---------------------------------------------------------------------------

BASELINE_FEATURES = [
    # demographic
    "_AGE_G", "_RACE1", "EDUCA", "INCOME3", "MARITAL", "EMPLOY1", "CHILDREN",
    # behavioral
    "EXERANY2", "SLEPTIM1", "SMOKE100", "alcohol_days_30", "BMI",
    # chronic
    "DIABETE4", "CVDCRHD4", "CVDSTRK3", "ASTHMA3",
    "CHCCOPD3", "HAVARTH4", "CHCKDNY2", "ADDEPEV3",
    "chronic_count",
    # healthcare access
    "PRIMINSR", "PERSDOC3", "MEDCOST1", "CHECKUP1",
]

ACE_FEATURES = [
    "ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN", "ACEDIVRC",
    "ACEPUNCH", "ACEHURT1", "ACESWEAR", "ACETOUCH", "ACETTHEM",
    "ACEHVSEX", "ACEADSAF", "ACEADNED",
    "ACE_score",
]

SDHE_FEATURES = [
    "LSATISFY", "EMTSUPRT", "SDHISOLT", "SDHEMPLY", "FOODSTMP",
    "SDHFOOD1", "SDHBILLS", "SDHUTILS", "SDHTRNSP", "SDHSTRE1",
    "SDHE_burden",
]

INTERACTION_FEATURES = ["ACE_x_SDHE"]

# Numeric features (receive StandardScaler)
ALL_NUMERIC = {
    "CHILDREN", "SLEPTIM1", "alcohol_days_30", "BMI", "chronic_count",
    "ACE_score", "SDHE_burden", "ACE_x_SDHE",
}

INCOME_COL = "INCOME3"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_configs(df_cols):
    raw = {
        "A_baseline": BASELINE_FEATURES,
        "B_ace":      BASELINE_FEATURES + ACE_FEATURES,
        "C_sdhe":     BASELINE_FEATURES + SDHE_FEATURES,
        "D_all":      BASELINE_FEATURES + ACE_FEATURES + SDHE_FEATURES + INTERACTION_FEATURES,
    }
    configs = {}
    for name, feats in raw.items():
        seen = set()
        deduped = []
        for f in feats:
            if f in df_cols and f not in seen:
                deduped.append(f)
                seen.add(f)
        configs[name] = deduped
    return configs


def get_feature_groups(feature_list):
    numeric     = [c for c in feature_list if c in ALL_NUMERIC]
    income      = [INCOME_COL] if INCOME_COL in feature_list else []
    categorical = [c for c in feature_list if c not in ALL_NUMERIC and c != INCOME_COL]
    return numeric, categorical, income


def build_preprocessor(feature_list):
    numeric, categorical, income = get_feature_groups(feature_list)
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    if income:
        transformers.append(("income", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
            ("onehot",  OneHotEncoder(handle_unknown="ignore")),
        ]), income))
    return ColumnTransformer(transformers=transformers)


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def tune_mlp(config_name, feature_list, X_train, y_train, X_val, y_val):
    """Grid-search MLP hyperparameters; return best params Series and full results DataFrame."""

    # Pre-process once to avoid redundant fitting across all combos
    preprocessor = build_preprocessor(feature_list)
    X_tr = preprocessor.fit_transform(X_train[feature_list], y_train)
    X_vl = preprocessor.transform(X_val[feature_list])

    # MLPClassifier has no class_weight param; use sample_weight instead
    sw_train = compute_sample_weight("balanced", y_train)

    # Hyperparameter grid (mirrors RQ1 MLP)
    hidden_layer_candidates = [
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (128, 64, 32),
    ]
    lr_candidates    = [0.0001, 0.001, 0.01]
    alpha_candidates = [0.0001, 0.001, 0.01]   # L2 regularisation strength

    total = len(hidden_layer_candidates) * len(lr_candidates) * len(alpha_candidates)
    print(f"\n{'='*70}")
    print(f"Tuning {total} MLP configs for {config_name}")
    print(f"{'='*70}")

    results = []
    start   = time.time()
    i       = 0

    for hidden_layers in hidden_layer_candidates:
        for lr in lr_candidates:
            for alpha in alpha_candidates:
                i += 1
                clf = MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    learning_rate_init=lr,
                    alpha=alpha,
                    max_iter=500,
                    early_stopping=True,
                    n_iter_no_change=30,
                    validation_fraction=0.1,
                    random_state=42,
                )
                clf.fit(X_tr, y_train, sample_weight=sw_train)

                y_pred = clf.predict(X_vl)
                y_prob = clf.predict_proba(X_vl)[:, 1]

                results.append({
                    "hidden_layers":    str(hidden_layers),
                    "lr":               lr,
                    "alpha":            alpha,
                    "n_iter":           clf.n_iter_,
                    "val_f1":           f1_score(y_val, y_pred, zero_division=0),
                    "val_balanced_acc": balanced_accuracy_score(y_val, y_pred),
                    "val_roc_auc":      roc_auc_score(y_val, y_prob),
                    "val_pr_auc":       average_precision_score(y_val, y_prob),
                    "val_recall":       recall_score(y_val, y_pred, zero_division=0),
                    "val_precision":    precision_score(y_val, y_pred, zero_division=0),
                })

                if i % 9 == 0 or i == total:
                    elapsed = time.time() - start
                    eta     = (total - i) * (elapsed / i)
                    best_so_far = max(r["val_f1"] for r in results)
                    print(f"  [{i}/{total}] Best F1={best_so_far:.4f} | ETA {eta/60:.1f}min")

    results_df = pd.DataFrame(results).sort_values("val_f1", ascending=False)
    elapsed = time.time() - start
    print(f"Done in {elapsed/60:.1f} min | Best val F1 = {results_df.iloc[0]['val_f1']:.4f}")

    return results_df.iloc[0], results_df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_on_test(config_name, feature_list, best_params, X_train_val, y_train_val, X_test, y_test):
    """Retrain on full train+val with best params, then evaluate on held-out test set."""

    hidden_layers = ast.literal_eval(best_params["hidden_layers"])
    lr            = float(best_params["lr"])
    alpha         = float(best_params["alpha"])

    preprocessor = build_preprocessor(feature_list)
    X_tv = preprocessor.fit_transform(X_train_val[feature_list], y_train_val)
    X_te = preprocessor.transform(X_test[feature_list])

    sw_tv = compute_sample_weight("balanced", y_train_val)

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        learning_rate_init=lr,
        alpha=alpha,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=30,
        validation_fraction=0.1,
        random_state=42,
    )
    clf.fit(X_tv, y_train_val, sample_weight=sw_tv)

    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    metrics = {
        "Config":       config_name,
        "Features":     len(feature_list),
        "hidden_layers": str(hidden_layers),
        "lr":           lr,
        "alpha":        alpha,
        "Accuracy":     accuracy_score(y_test, y_pred),
        "Balanced_Acc": balanced_accuracy_score(y_test, y_pred),
        "Precision":    precision_score(y_test, y_pred, zero_division=0),
        "Recall":       recall_score(y_test, y_pred, zero_division=0),
        "F1":           f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC":      roc_auc_score(y_test, y_prob),
        "PR_AUC":       average_precision_score(y_test, y_prob),
    }

    print(f"\n--- {config_name} Test Results ---")
    print(f"  hidden={hidden_layers}  lr={lr}  alpha={alpha}")
    print(f"  F1={metrics['F1']:.4f}  AUC={metrics['ROC_AUC']:.4f}  "
          f"Recall={metrics['Recall']:.4f}  Precision={metrics['Precision']:.4f}")

    return metrics, clf, preprocessor, y_pred, y_prob


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_comparison(results_table, configs, all_probs, y_test, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    config_labels = ["A: Baseline", "B: +ACE", "C: +SDHE", "D: +All"]
    colors = ["#95a5a6", "#9b59b6", "#e67e22", "#27ae60"]

    # Plot 1: F1
    f1_vals = results_table["F1"].values
    bars = axes[0].bar(config_labels, f1_vals, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("F1 Score by Feature Configuration", fontweight="bold")
    axes[0].set_ylabel("F1 Score")
    for i, (bar, val) in enumerate(zip(bars, f1_vals)):
        delta = val - f1_vals[0]
        label = f"{val:.3f}" if i == 0 else f"{val:.3f}\n(\u0394{delta:+.3f})"
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     label, ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, max(f1_vals) * 1.18)

    # Plot 2: ROC-AUC
    auc_vals = results_table["ROC_AUC"].values
    bars = axes[1].bar(config_labels, auc_vals, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_title("ROC-AUC by Feature Configuration", fontweight="bold")
    axes[1].set_ylabel("ROC-AUC")
    for i, (bar, val) in enumerate(zip(bars, auc_vals)):
        delta = val - auc_vals[0]
        label = f"{val:.3f}" if i == 0 else f"{val:.3f}\n(\u0394{delta:+.3f})"
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     label, ha="center", va="bottom", fontsize=9)
    axes[1].set_ylim(0, max(auc_vals) * 1.15)

    # Plot 3: ROC curves
    for name, color, label in zip(configs.keys(), colors, config_labels):
        RocCurveDisplay.from_predictions(
            y_test, all_probs[name], name=label,
            ax=axes[2], plot_chance_level=False, color=color)
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    axes[2].set_title("ROC Curves: All Configurations", fontweight="bold")
    axes[2].legend(fontsize=8, loc="lower right")

    plt.suptitle("RQ2: MLP -- Feature Augmentation Comparison",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(out_dir, "rq2_mlp_config_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Permutation Importance (Config D)
# ---------------------------------------------------------------------------

def plot_permutation_importance(clf, preprocessor, feature_list, X_test, y_test, out_dir, n_top=15):
    """Compute and plot permutation importance for Config D."""

    X_te = preprocessor.transform(X_test[feature_list])

    print("\nComputing permutation importance for Config D (this may take a moment)...")
    result = permutation_importance(
        clf, X_te, y_test,
        n_repeats=10,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
    )

    # Get feature names from preprocessor
    feature_names = preprocessor.get_feature_names_out()

    fi_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": result.importances_mean,
        "Std":        result.importances_std,
    }).sort_values("Importance", ascending=False)

    # Aggregate back to original feature names (collapse one-hot encoded columns)
    original_cols_sorted = sorted(feature_list, key=len, reverse=True)

    def get_original_feature(encoded_name):
        raw = encoded_name.split("__", 1)[1] if "__" in encoded_name else encoded_name
        for col in original_cols_sorted:
            if raw == col or raw.startswith(col + "_"):
                return col
        return raw

    def tag_source(name):
        name_upper = name.upper()
        if "ACE_X_SDHE" in name_upper:
            return "Interaction"
        if "ACE" in name_upper:
            return "ACE"
        sdhe_keys = ["LSATISFY", "EMTSUPRT", "SDHISOLT", "SDHEMPLY", "FOODSTMP",
                     "SDHFOOD", "SDHBILLS", "SDHUTILS", "SDHTRNSP", "SDHSTRE", "SDHE"]
        if any(k in name_upper for k in sdhe_keys):
            return "SDHE"
        return "Baseline"

    fi_df["Original_Feature"] = fi_df["Feature"].apply(get_original_feature)
    fi_df["Source"]           = fi_df["Original_Feature"].apply(tag_source)

    fi_agg = (
        fi_df.groupby("Original_Feature")
        .agg(
            Importance=("Importance", "sum"),
            Std=("Std", "mean"),
            Source=("Source", "first"),
        )
        .sort_values("Importance", ascending=False)
        .reset_index()
    )

    print(f"\nTop {n_top} features (Config D — Permutation Importance):")
    print(fi_agg.head(n_top).to_string(index=False))

    print("\nPermutation importance by source:")
    source_summary = fi_agg.groupby("Source")["Importance"].agg(["sum", "count"])
    source_summary["pct"] = source_summary["sum"] / source_summary["sum"].sum() * 100
    print(source_summary.round(3))

    # Plot
    top_n = fi_agg.head(n_top).copy()
    source_colors = {
        "Baseline":    "#95a5a6",
        "ACE":         "#9b59b6",
        "SDHE":        "#e67e22",
        "Interaction": "#27ae60",
    }
    bar_colors = [source_colors.get(s, "#bdc3c7") for s in top_n["Source"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(top_n) - 1, -1, -1)
    ax.barh(list(y_pos), top_n["Importance"].values, color=bar_colors,
            xerr=top_n["Std"].values, capsize=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top_n["Original_Feature"].values, fontsize=9)
    ax.set_xlabel("Mean decrease in F1 (Permutation Importance)")
    ax.set_title(
        f"RQ2 MLP: Top {n_top} Feature Importances\n(Config D: All Features)",
        fontweight="bold",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, label=l)
        for l, c in source_colors.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "rq2_mlp_permutation_importance_configD.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")

    return fi_agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Load data ---
    print(f"Loading data from: {DATA_PATH}")
    df_rq2 = pd.read_csv(DATA_PATH)
    assert df_rq2["target_fmd"].isnull().sum() == 0, "target_fmd has missing values!"
    print(f"RQ2 dataset: {df_rq2.shape}")
    print(f"\nTarget distribution:")
    print(df_rq2["target_fmd"].value_counts())
    print(df_rq2["target_fmd"].value_counts(normalize=True))

    # --- Build configs ---
    configs = build_configs(df_rq2.columns)
    print("\nFeature configurations:")
    for name, cols in configs.items():
        print(f"  {name}: {len(cols)} features")

    print(f"\nNew features in B vs A: {set(configs['B_ace']) - set(configs['A_baseline'])}")
    print(f"New features in C vs A: {set(configs['C_sdhe']) - set(configs['A_baseline'])}")
    print(f"New features in D vs A: {set(configs['D_all']) - set(configs['A_baseline'])}")

    for name, cols in configs.items():
        num, cat, inc = get_feature_groups(cols)
        print(f"{name}: {len(num)} numeric, {len(cat)} categorical, {len(inc)} income")

    # --- Split ---
    X_all = df_rq2.drop(columns=["target_fmd"])
    y_all = df_rq2["target_fmd"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

    print(f"\nTrain: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")
    print(f"Train pos rate: {y_train.mean():.3f}")
    print(f"Val pos rate:   {y_val.mean():.3f}")
    print(f"Test pos rate:  {y_test.mean():.3f}")

    # --- Run all 4 configs ---
    all_best_params    = {}
    all_tuning_results = {}
    all_test_metrics   = []
    all_models         = {}
    all_preds          = {}
    all_probs          = {}

    for config_name, feature_list in configs.items():
        best_params, tuning_df = tune_mlp(
            config_name, feature_list, X_train, y_train, X_val, y_val)
        all_best_params[config_name]    = best_params
        all_tuning_results[config_name] = tuning_df

        metrics, clf, preprocessor, y_pred, y_prob = eval_on_test(
            config_name, feature_list, best_params,
            X_train_val, y_train_val, X_test, y_test)

        all_test_metrics.append(metrics)
        all_models[config_name] = (clf, preprocessor)
        all_preds[config_name]  = y_pred
        all_probs[config_name]  = y_prob

    print("\n" + "="*70)
    print("All 4 configurations completed!")
    print("="*70)

    # --- Results table ---
    results_table = pd.DataFrame(all_test_metrics)

    baseline_f1  = results_table.loc[results_table["Config"] == "A_baseline", "F1"].values[0]
    baseline_auc = results_table.loc[results_table["Config"] == "A_baseline", "ROC_AUC"].values[0]

    results_table["\u0394F1"]  = results_table["F1"]      - baseline_f1
    results_table["\u0394AUC"] = results_table["ROC_AUC"] - baseline_auc

    results_table["\u0394F1_str"]  = results_table["\u0394F1"].apply(
        lambda x: f"+{x:.4f}" if x > 0 else (f"{x:.4f}" if x != 0 else "-"))
    results_table["\u0394AUC_str"] = results_table["\u0394AUC"].apply(
        lambda x: f"{x:.4f}" if x > 0 else (f"{x:.4f}" if x != 0 else "-"))

    print("\n" + "="*60)
    print("RQ2 Results: MLP -- Feature Augmentation Comparison")
    print("="*60)
    print(results_table[[
        "Config", "Features", "hidden_layers", "lr", "alpha",
        "F1", "\u0394F1_str", "ROC_AUC", "\u0394AUC_str",
        "Recall", "Precision", "Balanced_Acc", "PR_AUC",
    ]].round(4).to_string(index=False))

    b_delta = results_table.loc[results_table["Config"] == "B_ace",  "\u0394F1"].values[0]
    c_delta = results_table.loc[results_table["Config"] == "C_sdhe", "\u0394F1"].values[0]
    d_delta = results_table.loc[results_table["Config"] == "D_all",  "\u0394F1"].values[0]

    print(f"\n\u0394F1 (B vs A):  {b_delta:.4f}  <- ACE alone")
    print(f"\u0394F1 (C vs A):  {c_delta:.4f}  <- SDHE alone")
    print(f"\u0394F1 (D vs A):  {d_delta:.4f}  <- ACE + SDHE + interaction")

    # --- Classification reports ---
    for name in configs.keys():
        print(f"\n{'='*60}")
        print(f"Config {name} -- Classification Report")
        print(f"{'='*60}")
        print(classification_report(y_test, all_preds[name]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, all_preds[name]))

    # --- Best params summary ---
    print("\nBest hyperparameters selected per configuration (by val F1):\n")
    for name, params in all_best_params.items():
        print(f"  {name}:")
        print(f"    hidden_layers    = {params['hidden_layers']}")
        print(f"    lr               = {params['lr']}")
        print(f"    alpha            = {params['alpha']}")
        print(f"    val_f1           = {params['val_f1']:.4f}")
        print()

    # --- Visualization ---
    plot_comparison(results_table, configs, all_probs, y_test, SAVE_DIR)

    # --- Permutation Importance: Config D ---
    clf_d, preprocessor_d = all_models["D_all"]
    fi_agg = plot_permutation_importance(
        clf_d, preprocessor_d, configs["D_all"], X_test, y_test, SAVE_DIR)
    fi_agg.to_csv(os.path.join(SAVE_DIR, "rq2_mlp_permutation_importance_configD.csv"), index=False)

    # --- Save CSVs ---
    results_table.to_csv(os.path.join(SAVE_DIR, "rq2_mlp_comparison.csv"), index=False)
    for name, df in all_tuning_results.items():
        df.to_csv(os.path.join(SAVE_DIR, f"rq2_mlp_tuning_{name}.csv"), index=False)

    # --- Final summary ---
    print("="*70)
    print("RQ2 MLP -- FINAL SUMMARY")
    print("="*70)
    for _, row in results_table.iterrows():
        print(f"\n  Config {row['Config']}:")
        print(f"    Features:      {int(row['Features'])}")
        print(f"    hidden_layers: {row['hidden_layers']}")
        print(f"    lr={row['lr']}  alpha={row['alpha']}")
        print(f"    F1={row['F1']:.4f}  AUC={row['ROC_AUC']:.4f}  \u0394F1={row[chr(0x394)+'F1']:+.4f}")

    best_config = results_table.loc[results_table["F1"].idxmax()]
    print(f"\nBest config: {best_config['Config']} (F1={best_config['F1']:.4f})")
    print(f"\nAll results saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()