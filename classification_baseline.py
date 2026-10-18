import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "1792329196594_WA_Fn-UseC_-Telco-Customer-Churn.csv"
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
print("Blank/invalid TotalCharges:", int(df["TotalCharges"].isna().sum()))
print("TotalCharges dtype after fix:", df["TotalCharges"].dtype)

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
print("Churn rate:", round(df["Churn"].mean(), 4))

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("Churn rate train:", round(y_train.mean(), 4), "| test:", round(y_test.mean(), 4))

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
binary_numeric = ["SeniorCitizen"]
scale_cols = [c for c in numeric_cols if c not in binary_numeric]

print("Numeric (scaled):", scale_cols)
print("Passthrough binary:", binary_numeric)
print("Categorical (one-hot):", len(categorical_cols), "columns")


def build_pipeline(class_weight=None):
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    pre = ColumnTransformer(
        [
            ("num", numeric_pipe, scale_cols),
            ("bin", "passthrough", binary_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=RANDOM_STATE)),
        ]
    )


models = {
    "Task 2 Baseline (majority class)": Pipeline(
        [("clf", DummyClassifier(strategy="most_frequent"))]
    ),
    "LogReg (default weights)": build_pipeline(None),
    "LogReg (class_weight='balanced')": build_pipeline("balanced"),
}

results = {}
matrices = {}
probas = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    matrices[name] = confusion_matrix(y_test, pred)
    probas[name] = proba

    print("\n" + "=" * 60)
    print(name)
    print(classification_report(y_test, pred, target_names=["No churn", "Churn"], zero_division=0))
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(matrices[name])

metrics_df = pd.DataFrame(results).T.round(4)
print("\n" + "=" * 60)
print(metrics_df.to_string())

thresholds = [0.30, 0.40, 0.50]
threshold_rows = []
for name in list(models)[1:]:
    for t in thresholds:
        pred_t = (probas[name] >= t).astype(int)
        cm = confusion_matrix(y_test, pred_t)
        threshold_rows.append(
            {
                "model": name,
                "threshold": t,
                "precision": round(precision_score(y_test, pred_t, zero_division=0), 4),
                "recall": round(recall_score(y_test, pred_t, zero_division=0), 4),
                "f1": round(f1_score(y_test, pred_t, zero_division=0), 4),
                "FN": int(cm[1, 0]),
                "FP": int(cm[0, 1]),
            }
        )
threshold_df = pd.DataFrame(threshold_rows)
print("\nThreshold sweep")
print(threshold_df.to_string(index=False))

metrics_df.to_csv("metrics_table.csv")
threshold_df.to_csv("threshold_sweep.csv", index=False)
with open("results.json", "w") as f:
    json.dump(
        {
            "metrics": results,
            "confusion_matrices": {k: v.tolist() for k, v in matrices.items()},
        },
        f,
        indent=2,
    )

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, cm) in zip(axes, matrices.items()):
    ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(name, fontsize=10)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()

lr_balanced = models["LogReg (class_weight='balanced')"]
feature_names = lr_balanced.named_steps["pre"].get_feature_names_out()
coefs = pd.Series(lr_balanced.named_steps["clf"].coef_[0], index=feature_names).sort_values()
print("\nTop churn-increasing features (balanced model):")
print(coefs.tail(5).round(3).to_string())
print("Top churn-reducing features (balanced model):")
print(coefs.head(5).round(3).to_string())
print("\nDone. Saved metrics_table.csv, threshold_sweep.csv, results.json, confusion_matrices.png")
