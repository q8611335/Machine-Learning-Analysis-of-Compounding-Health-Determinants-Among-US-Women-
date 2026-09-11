import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A ── 固定參數（不進入搜尋，直接改這裡套用到所有實驗）
# ══════════════════════════════════════════════════════════════════════════════
FIXED = {
    "epochs":         100,
    "batch_size":     256,
    "activation":     "relu",    # "relu" | "tanh" | "leaky_relu" | "elu"
    "batch_norm":     True,
    "optimizer":      "adam",    # "adam" | "adamw" | "sgd"
    "weight_decay":   1e-4,
    "class_weight":   True,
    "early_stopping": 15,        # 幾個 epoch 沒進步就停（0 = 不用）
    "lr_scheduler":   "reduce_on_plateau",  # "none" | "step" | "cosine" | "reduce_on_plateau"
    "lr_gamma":       0.5,
    "seed":           42,
}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B ── 搜尋範圍（每個 list 裡列出想試的值，程式會自動跑全部組合）
# ══════════════════════════════════════════════════════════════════════════════
SEARCH_SPACE = {
    "hidden_layers": [[64], [128], [128, 64], [128, 64, 32]],
    "dropout":       [0.0, 0.2, 0.3],
    "learning_rate": [1e-3, 5e-4],
}
# 總組合數 = 4 × 3 × 2 = 24 組

# ══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

ACTIVATIONS = {
    "relu":       nn.ReLU(),
    "tanh":       nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(0.1),
    "elu":        nn.ELU(),
    "logistic":   nn.Sigmoid(),
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ── 1. Load data ──────────────────────────────────────────────────────────────
df_rq1 = pd.read_csv("df_rq1_baseline.csv")
print("Dataset shape:", df_rq1.shape)
assert df_rq1["target_fmd"].isnull().sum() == 0, "target_fmd has missing values!"

# ── 2. Split features and target ─────────────────────────────────────────────
X = df_rq1.drop(columns=["target_fmd"])
y = df_rq1["target_fmd"]

print("y distribution:")
print(y.value_counts(normalize=True))

# ── 3. Three-way split: 60% train / 20% val / 20% test ───────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=FIXED["seed"], stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=FIXED["seed"], stratify=y_temp
)

print(f"\nTrain      : {X_train.shape[0]:>6} rows ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Validation : {X_val.shape[0]:>6} rows ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"Test       : {X_test.shape[0]:>6} rows ({X_test.shape[0]/len(X)*100:.0f}%)")

# ── 4. Define feature groups ──────────────────────────────────────────────────
numeric_features = [
    "CHILDREN", "SLEPTIM1", "alcohol_days_30", "BMI", "chronic_count",
]
income_feature = ["INCOME3"]
categorical_features = [
    col for col in X.columns if col not in numeric_features + income_feature
]

# ── 5. Preprocessing ──────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical_features),
    ("income", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), income_feature),
])

X_train_np = preprocessor.fit_transform(X_train).astype(np.float32)
X_val_np   = preprocessor.transform(X_val).astype(np.float32)
X_test_np  = preprocessor.transform(X_test).astype(np.float32)
INPUT_DIM  = X_train_np.shape[1]


def make_dataset(X_np, y_series):
    X_t = torch.tensor(X_np,            dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_series.values, dtype=torch.long).to(DEVICE)
    return TensorDataset(X_t, y_t)


train_ds = make_dataset(X_train_np, y_train)
val_ds   = make_dataset(X_val_np,   y_val)
test_ds  = make_dataset(X_test_np,  y_test)

# ── 6. Model builder ──────────────────────────────────────────────────────────
def build_mlp(params: dict, input_dim: int) -> nn.Sequential:
    act_fn = ACTIVATIONS[params["activation"]]
    layers = []
    in_dim = input_dim

    for out_dim in params["hidden_layers"]:
        layers.append(nn.Linear(in_dim, out_dim))
        if params.get("batch_norm"):
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(act_fn)
        if params.get("dropout", 0) > 0:
            layers.append(nn.Dropout(params["dropout"]))
        in_dim = out_dim

    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)


# ── 7. Training function ──────────────────────────────────────────────────────
def train_model(params: dict, input_dim: int, verbose: bool = True):
    set_seed(params["seed"])

    loader_train = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    loader_val   = DataLoader(val_ds,   batch_size=params["batch_size"])

    if params.get("class_weight"):
        counts    = np.bincount(y_train.values.astype(int))
        weights   = torch.tensor(len(y_train) / (2 * counts), dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    model = build_mlp(params, input_dim).to(DEVICE)
    optimizer = {
        "adam":  torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd":   torch.optim.SGD,
    }[params["optimizer"]](
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params.get("weight_decay", 0),
    )

    sched_name = params.get("lr_scheduler", "none")
    if sched_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=params["lr_gamma"],
            patience=max(1, params["early_stopping"] // 2),
        )
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=params["lr_gamma"])
    elif sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])
    else:
        scheduler = None

    best_score = -1.0
    best_state = None
    no_improve = 0
    patience   = params.get("early_stopping", 0)

    for epoch in range(1, params["epochs"] + 1):
        model.train()
        for X_b, y_b in loader_train:
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X_b, y_b in loader_val:
                preds.extend(model(X_b).argmax(dim=1).cpu().numpy())
                labels.extend(y_b.cpu().numpy())

        val_score = balanced_accuracy_score(labels, preds)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_score)
        elif scheduler is not None:
            scheduler.step()

        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch:>4}  val_balanced_acc={val_score:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, best_score


# ── 8. 自動全排列搜尋 ─────────────────────────────────────────────────────────
keys   = list(SEARCH_SPACE.keys())
values = list(SEARCH_SPACE.values())
combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

total = len(combos)
print(f"\n── Hyperparameter search：共 {total} 組組合 ──")
print(f"{'#':>3}  {'hidden_layers':<22} {'dropout':>7} {'lr':>8}  {'val_balanced_acc':>17}")
print("─" * 66)

results        = []
best_params    = None
best_val_score = -1.0

for i, overrides in enumerate(combos, start=1):
    params = {**FIXED, **overrides}
    _, score = train_model(params, INPUT_DIM, verbose=False)
    results.append((score, params))

    marker = " ◀" if score > best_val_score else ""
    print(f"{i:>3}  {str(params['hidden_layers']):<22} {params['dropout']:>7.1f} "
          f"{params['learning_rate']:>8.0e}  {score:>17.4f}{marker}")

    if score > best_val_score:
        best_val_score = score
        best_params    = params

# 排序結果方便查看
results.sort(key=lambda x: x[0], reverse=True)
print(f"\nTop 3 組合：")
for rank, (score, p) in enumerate(results[:3], start=1):
    print(f"  #{rank}  layers={p['hidden_layers']}  dropout={p['dropout']}  "
          f"lr={p['learning_rate']}  → {score:.4f}")

print(f"\nBest → layers={best_params['hidden_layers']}  dropout={best_params['dropout']}  "
      f"lr={best_params['learning_rate']}  (val balanced_accuracy={best_val_score:.4f})")

# ── 9. Train final model with best params ────────────────────────────────────
print("\n── Training final model ──")
final_model, _ = train_model(best_params, INPUT_DIM, verbose=True)

# ── 10. Final evaluation on TEST set (only once) ─────────────────────────────
print("\n── Final evaluation on TEST set ──")
final_model.eval()
loader_test = DataLoader(test_ds, batch_size=FIXED["batch_size"])

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for X_b, y_b in loader_test:
        logits = final_model(X_b)
        all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y_b.cpu().numpy())

y_pred = np.array(all_preds)
y_prob = np.array(all_probs)
y_true = np.array(all_labels)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print(f"\nBalanced Accuracy : {balanced_accuracy_score(y_true, y_pred):.4f}")
print(f"ROC AUC Score     : {roc_auc_score(y_true, y_prob):.4f}")
