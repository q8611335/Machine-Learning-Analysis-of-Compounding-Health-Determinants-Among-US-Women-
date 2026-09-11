import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier

# ── 1. Load data ──────────────────────────────────────────────────────────────
df_rq1 = pd.read_csv("df_rq1_baseline.csv")
print("Dataset shape:", df_rq1.shape)

assert df_rq1["target_fmd"].isnull().sum() == 0, "target_fmd has missing values!"

# ── 2. Split features and target ─────────────────────────────────────────────
X = df_rq1.drop(columns=["target_fmd"])
y = df_rq1["target_fmd"]

print("y distribution:")
print(y.value_counts(normalize=True))

# ── 3. Three-way split: 60% train / 20% validation / 20% test ────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)

print(f"\nTrain      : {X_train.shape[0]:>6} rows ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Validation : {X_val.shape[0]:>6} rows ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"Test       : {X_test.shape[0]:>6} rows ({X_test.shape[0]/len(X)*100:.0f}%)")

# ── 4. Define feature groups ──────────────────────────────────────────────────
numeric_features = [
    "CHILDREN",
    "SLEPTIM1",
    "alcohol_days_30",
    "BMI",
    "chronic_count",
]

income_feature = ["INCOME3"]

categorical_features = [
    col for col in X.columns if col not in numeric_features + income_feature
]

# ── 5. Preprocessing pipeline ────────────────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

income_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num",    numeric_transformer,     numeric_features),
    ("cat",    categorical_transformer, categorical_features),
    ("income", income_transformer,      income_feature),
])
# 診斷用：不做任何 class_weight 調整，看看預測分佈
diag_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )),
])

# checkpoint
# diag_model.fit(X_train, y_train)
# y_diag_pred = diag_model.predict(X_val)
# print("預測分佈:", pd.Series(y_diag_pred).value_counts())
# print("Balanced Accuracy:", balanced_accuracy_score(y_val, y_diag_pred))
# print("ROC AUC:", roc_auc_score(y_val, diag_model.predict_proba(X_val)[:, 1]))


# ── 6. Hyperparameter search on VALIDATION set ───────────────────────────────
# Tune n_estimators: number of trees in the forest.
# More trees = more stable predictions, but slower to train.

print("\n── Hyperparameter search (validation set) ──")
best_n_estimators = None
best_val_score = -1

for n_estimators in [300, 400, 500, 600]:
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,          # use all CPU cores to speed up training
        )),
    ])
    model.fit(X_train, y_train)
    val_score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"  n_estimators={n_estimators:<4} → val roc_auc = {val_score:.4f}")

    if val_score > best_val_score:
        best_val_score = val_score
        best_n_estimators = n_estimators

print(f"\nBest n_estimators: {best_n_estimators}  (val roc_auc = {best_val_score:.4f})")

# ── 7. Train final model with best hyperparameters ───────────────────────────
final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=best_n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])
final_model.fit(X_train, y_train)

# ── 8. Final evaluation on TEST set (only once) ──────────────────────────────
print("\n── Final evaluation on TEST set ──")
y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nBalanced Accuracy : {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score     : {roc_auc_score(y_test, y_prob):.4f}")