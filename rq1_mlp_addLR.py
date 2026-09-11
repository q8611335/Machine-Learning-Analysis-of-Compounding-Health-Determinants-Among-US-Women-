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
from sklearn.neural_network import MLPClassifier

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
# Tune hidden_layer_sizes: a tuple defines the number of layers and neurons.
# e.g. (64,) = one hidden layer with 64 neurons
#      (64, 32) = two hidden layers with 64 then 32 neurons

print("\n── Hyperparameter search (validation set) ──")
best_hidden_layers = None
best_lr = None
best_val_score = -1

hidden_layer_candidates = [
    (64,),
    (128,),
    (64, 32),
    (128, 64),
    (128, 64, 32),
]

lr_candidates = [0.001, 0.01, 0.0001]

for hidden_layers in hidden_layer_candidates:
    for lr in lr_candidates:
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation="relu",
                learning_rate_init=lr,
                max_iter=1000,
                random_state=42,
            )),
        ])
        model.fit(X_train, y_train)
        val_score = balanced_accuracy_score(y_val, model.predict(X_val))
        print(f"  hidden_layer_sizes={str(hidden_layers):<15} lr={lr} → val balanced_accuracy = {val_score:.4f}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_hidden_layers = hidden_layers
            best_lr = lr

print(f"\nBest hidden_layer_sizes: {best_hidden_layers}, best lr: {best_lr}  (val balanced_accuracy = {best_val_score:.4f})")

# ── 7. Train final model with best hyperparameters ───────────────────────────
final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=best_hidden_layers,
        activation="relu",
        learning_rate_init=best_lr,
        max_iter=1000,
        random_state=42,
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