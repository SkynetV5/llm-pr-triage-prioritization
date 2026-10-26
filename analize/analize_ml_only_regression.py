import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    ndcg_score,
)
from baseline_fifo import calculate_mrr
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(
    "github_prs_ground_truth_sample_2000.csv",
    index_col=0,
    parse_dates=["created_at", "closed_at"],
)

df["title_len"] = df["pr_title"].fillna("").apply(lambda x: len(str(x)))
df["body_len"] = df["pr_body"].fillna("").apply(lambda x: len(str(x)))

df["has_tests"] = df["has_tests"].fillna("").apply(lambda x: 1 if x == True else 0)

feature_cols = [
    "additions",
    "deletions",
    "changed_files",
    "directories_count",
    "diff_lines",
    "commits_count",
    "title_len",
    "body_len",
    "has_tests",
]

additional_cols = [
    "author_accept_rate",
    "is_core_member",
    "author_past_prs",
]

for col in additional_cols:
    if col in df.columns:
        feature_cols.append(col)


for col in feature_cols:
    df[col] = df[col].fillna(0)


df["target"] = df["gt_class_num"]

df = df.sort_values("created_at")

train_df, test_df = [], []


split_idx = int(len(df) * 0.8)
train_df.append(df.iloc[:split_idx])
test_df.append(df.iloc[split_idx:])


train_data = pd.concat(train_df).reset_index(drop=True)
test_data = pd.concat(test_df).reset_index(drop=True)


X_train, y_train = train_data[feature_cols], train_data["target"]
X_test, y_test = test_data[feature_cols], test_data["target"]

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    bootstrap=True,
)
model.fit(X_train, y_train)

test_data["ml_pred_score"] = model.predict(X_test)

test_data["ml_pred_class"] = np.clip(np.round(test_data["ml_pred_score"]), 0, 3).astype(
    int
)

class_map_rev = {3: "P0", 2: "P1", 1: "P2", 0: "P3"}

test_data["ml_pred_priority_text"] = test_data["ml_pred_class"].map(class_map_rev)


mrr_list = []
ndcg5_list = []
ndcg10_list = []
spearman_list = []

for repo_name, group in test_data.groupby("repo"):

    if len(group) < 3:
        continue

    y_true = group["target"].values
    y_pred = group["ml_pred_score"].values

    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        spearmanr_score = spearmanr(y_true, y_pred)[0]
        if not np.isnan(spearmanr_score):
            spearman_list.append(spearmanr_score)

    mrr_score = calculate_mrr(y_true, y_pred)
    mrr_list.append(mrr_score)

    ndcg_score_5 = ndcg_score([y_true], [y_pred], k=5)
    ndcg5_list.append(ndcg_score_5)

    ndcg_score_10 = ndcg_score([y_true], [y_pred], k=10)
    ndcg10_list.append(ndcg_score_10)

class_counts = test_data["ml_pred_priority_text"].value_counts().to_dict()


results_ml = {
    "Wariant": "ML-Only Regression (Random Forest)",
    "NDCG@5": round(np.mean(ndcg5_list), 4),
    "NDCG@10": round(np.mean(ndcg10_list), 4),
    "MRR": round(np.mean(mrr_list), 4),
    "Spearman_Score": round(np.mean(spearman_list), 4),
    "P0_count": class_counts.get("P0", 0),
    "P1_count": class_counts.get("P1", 0),
    "P2_count": class_counts.get("P2", 0),
    "P3_count": class_counts.get("P3", 0),
    "P0_percent": round((class_counts.get("P0", 0) / len(test_data)) * 100, 2),
    "P1_percent": round((class_counts.get("P1", 0) / len(test_data)) * 100, 2),
    "P2_percent": round((class_counts.get("P2", 0) / len(test_data)) * 100, 2),
    "P3_percent": round((class_counts.get("P3", 0) / len(test_data)) * 100, 2),
}


summary_ml_df = pd.DataFrame([results_ml])
print(summary_ml_df.to_string(index=False))

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
    ascending=False
)

print(importances.round(4).to_string())
