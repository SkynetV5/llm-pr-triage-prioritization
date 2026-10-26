import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


def add_time_urgency_ground_truth(df):
    df_gt = df.copy()

    # Log time
    df_gt["log_duration"] = np.log1p(df_gt["duration_hours"])

    mean_per_repo = df_gt.groupby("repo")["log_duration"].transform("mean")
    std_per_repo = df_gt.groupby("repo")["log_duration"].transform("std")

    std_per_repo = std_per_repo.replace(0, np.nan).fillna(1.0)

    # Z-Score
    df_gt["norm_duration"] = (df_gt["log_duration"] - mean_per_repo) / std_per_repo

    pct_ranks = df_gt["norm_duration"].rank(method="first", pct=True)

    df_gt["gt_class_num"] = pd.cut(
        pct_ranks,
        bins=[0.0, 0.25, 0.50, 0.75, 1.00],
        labels=[3, 2, 1, 0],
        include_lowest=True,
    ).astype(int)

    class_map = {
        3: "P0 (Krytyczny)",
        2: "P1 (Wysoki)",
        1: "P2 (Średni)",
        0: "P3 (Niski)",
    }
    df_gt["gt_p_class"] = df_gt["gt_class_num"].map(class_map)

    return df_gt


if __name__ == "__main__":

    valid_sample = pd.read_csv(
        "cleaned_dataset.csv", index_col=0, parse_dates=["created_at", "closed_at"]
    )

    valid_sample_gt = add_time_urgency_ground_truth(valid_sample)

    print("--- ROZKŁAD KLAS GROUND TRUTH ---")
    print(valid_sample_gt["gt_p_class"].value_counts().sort_index())

    valid_sample_gt_2000 = valid_sample_gt.sample(n=2000, random_state=42).copy()

    # for idx,row in valid_sample_gt_2000.iterrows():
    #     print(row["labels"])

    valid_sample_gt_2000.to_csv("github_prs_ground_truth_sample_2000.csv", index=True)
