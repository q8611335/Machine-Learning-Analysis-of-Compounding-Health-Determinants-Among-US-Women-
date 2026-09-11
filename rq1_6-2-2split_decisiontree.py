import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.tree import DecisionTreeClassifier

# ── 1. Load data ──────────────────────────────────────────────────────────────
df_rq1 = pd.read_csv("df_rq1_baseline.csv")
print("Dataset shape:", df_rq1.shape)

# Check target has no missing values
assert df_rq1["target_fmd"].isnull().sum() == 0, "target_fmd has missing values!"

# ── 2. Split features and target ─────────────────────────────────────────────
X = df_rq1.drop(columns=["target_fmd"])
y = df_rq1["target_fmd"]

print("y distribution:")
print(y.value_counts(normalize=True))

# ── 3. Three-way split: 60% train / 20% validation / 20% test ────────────────
# Step 1: split off 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y,
)

# Step 2: split temp into 50/50 → validation 20%, test 20%
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
# Numeric  : median imputation + standardisation
# Categorical : most-frequent imputation + one-hot encoding
# INCOME3  : treat missing as its own category (-1) + one-hot encoding

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore")),
])

income_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
    ("onehot",  OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num",    numeric_transformer,     numeric_features),
    ("cat",    categorical_transformer, categorical_features),
    ("income", income_transformer,      income_feature),
])

# ── 6. Hyperparameter search on VALIDATION set ───────────────────────────────
# Tune max_depth; keep other settings fixed.
# Only X_val / y_val is used here — X_test is never touched.

print("\n── Hyperparameter search (validation set) ──")
best_depth = None
best_val_score = -1

for depth in [3, 5, 8, 10, 15]:
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            criterion="gini",
            max_depth=depth,
            min_samples_split=50,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    val_score = balanced_accuracy_score(y_val, model.predict(X_val))
    print(f"  max_depth={depth:>2} → val balanced_accuracy = {val_score:.4f}")

    if val_score > best_val_score:
        best_val_score = val_score
        best_depth = depth

print(f"\nBest max_depth: {best_depth}  (val balanced_accuracy = {best_val_score:.4f})")

# ── 7. Train final model with best hyperparameters ───────────────────────────
final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion="gini",
        max_depth=best_depth,
        min_samples_split=50,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )),
])
final_model.fit(X_train, y_train)

# ── 8. Final evaluation on TEST set (only once) ───────────────────────────────
print("\n── Final evaluation on TEST set ──")
y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nBalanced Accuracy : {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score     : {roc_auc_score(y_test, y_prob):.4f}")