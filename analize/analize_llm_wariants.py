import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from baseline_fifo import calculate_mrr
import numpy as np


def evaluate_variant(df, pred_col, name="Wariant"):

    mrr_list = []
    ndcg5_list = []
    ndcg10_list = []
    spearman_list = []

    for repo_name, group in df.groupby("repo"):

        if len(group) < 3:
            continue

        y_true = group["gt_class_num"].values
        y_pred = group[pred_col].values

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

    class_counts = (
        df[pred_col.replace("llm_pred_class_num", "llm_pred_priority_text")]
        .value_counts()
        .to_dict()
    )

    return {
        "Wariant": name,
        "NDCG@5": round(np.mean(ndcg5_list), 4),
        "NDCG@10": round(np.mean(ndcg10_list), 4),
        "MRR": round(np.mean(mrr_list), 4),
        "Spearman_Score": round(np.mean(spearman_list), 4),
        "P0_count": class_counts.get("P0", 0),
        "P1_count": class_counts.get("P1", 0),
        "P2_count": class_counts.get("P2", 0),
        "P3_count": class_counts.get("P3", 0),
        "P0_percent": (class_counts.get("P0", 0) / len(df)) * 100,
        "P1_percent": (class_counts.get("P1", 0) / len(df)) * 100,
        "P2_percent": (class_counts.get("P2", 0) / len(df)) * 100,
        "P3_percent": (class_counts.get("P3", 0) / len(df)) * 100,
    }


if __name__ == "__main__":

    var_a = pd.read_csv(
        "github_prs_llm_wariant_a.csv",
        index_col=0,
        parse_dates=["created_at", "closed_at"],
    )
    var_b = pd.read_csv(
        "github_prs_llm_wariant_b.csv",
        index_col=0,
        parse_dates=["created_at", "closed_at"],
    )
    var_c = pd.read_csv(
        "github_prs_llm_wariant_c.csv",
        index_col=0,
        parse_dates=["created_at", "closed_at"],
    )
    var_d = pd.read_csv(
        "github_prs_llm_wariant_d.csv",
        index_col=0,
        parse_dates=["created_at", "closed_at"],
    )
    # print("--- ROZKŁAD KLAS Klasyfikacji LLM - Wariant A (Title + Description) ---")
    # print(var_a["llm_pred_priority_text"].value_counts().sort_index())

    # print(
    #     "--- ROZKŁAD KLAS Klasyfikacji LLM - Wariant B (Title + Description + Metadata) ---"
    # )
    # print(var_b["llm_pred_priority_text"].value_counts().sort_index())

    # print(
    #     "--- ROZKŁAD KLAS Klasyfikacji LLM - Wariant C (Title + Description + Code diff) ---"
    # )
    # print(var_c["llm_pred_priority_text"].value_counts().sort_index())

    # print(
    #     "--- ROZKŁAD KLAS Klasyfikacji LLM - Wariant D (Title + Description + Metadata + Code diff) ---"
    # )
    # print(var_d["llm_pred_priority_text"].value_counts().sort_index())

    # print("--- ROZKŁAD KLAS Ground Truth ---")
    # print(var_a["gt_p_class"].value_counts().sort_index())

    results_variants = []

    results_variants.append(
        evaluate_variant(var_a, "llm_pred_class_num", "Wariant A (Title + Description)")
    )
    results_variants.append(
        evaluate_variant(
            var_b, "llm_pred_class_num", "Wariant B (Title + Description + Metadata)"
        )
    )
    results_variants.append(
        evaluate_variant(
            var_c, "llm_pred_class_num", "Wariant C (Title + Description + Code diff)"
        )
    )
    results_variants.append(
        evaluate_variant(
            var_d,
            "llm_pred_class_num",
            "Wariant D (Title + Description + Metadata + Code diff)",
        )
    )

    summary_df = pd.DataFrame(results_variants)
    print(summary_df.to_string(index=False))
