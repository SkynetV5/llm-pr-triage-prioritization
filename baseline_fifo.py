import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

def calculate_mrr(y_true,y_score):
    max_y_true = np.max(y_true)
    
    sorted_y_score = np.argsort(y_score)[::-1]
    
    for rank, idx in enumerate(sorted_y_score, start=1):
        if y_true[idx] == max_y_true:
            return 1.0 / rank
    return 0.0
    


valid_sample_gt = pd.read_csv("github_prs_ground_truth_sample_2000.csv", index_col=0, parse_dates=["created_at", "closed_at"])

ndcg5_list = []
ndcg10_list = []
mrr_list = []
spearman_list = []


for repo_name, group in valid_sample_gt.groupby("repo"):

    if len(group) < 3: continue

    y_true = group["gt_class_num"].values
    y_score = -group["created_at"].rank(method="first").values

    ndcg5_list.append(ndcg_score([y_true],[y_score], k=5))
    ndcg10_list.append(ndcg_score([y_true],[y_score], k=10))
    mrr_list.append(calculate_mrr(y_true,y_score))
    spearman_list.append(spearmanr(y_true,y_score)[0])
    

ndcg5_mean = round(np.nanmean(ndcg5_list),3)
ndcg10_mean = round(np.nanmean(ndcg10_list),3)
mrr_mean = round(np.nanmean(mrr_list),3)
spearman_mean = round(np.nanmean(spearman_list),3)

print(ndcg5_mean) # 0.665
print(ndcg10_mean) # 0.713
print(mrr_mean) # 0.568
print(spearman_mean) # -0.202

# Wyglada na to ze kolejka według czasu utworzenia działą 
# wręcz odwroteni do rzeczywistego czasu priorytetu
# spycha świeże, krytyczne poprawki na koniec/środek kolejki

