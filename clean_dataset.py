import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter
import re


def print_json_tree(data, indent="", is_last=True, max_depth=3, current_depth=0):
    """Rekurencyjnie wypisuje drzewo kluczy json"""
    if current_depth > max_depth:
        return

    marker = "└── " if is_last else "├── "

    if isinstance(data, dict):
        keys = list(data.keys())
        for idx, key in enumerate(keys):
            last_item = idx == len(keys) - 1
            val = data[key]
            val_type = type(val).__name__

            if isinstance(val, dict):
                print(f"{indent}{marker}{key} (dict, {len(val)} keys)")
                print_json_tree(
                    val,
                    indent + ("    " if last_item else "│   "),
                    last_item,
                    max_depth,
                    current_depth + 1,
                )
            elif isinstance(val, list):
                element_type = type(val[0]).__name__ if val else "empty"
                print(
                    f"{indent}{marker}📋 {key} (list[{element_type}], len={len(val)})"
                )

                # Jeśli lista zawiera słowniki, wyświetlamy strukturę pierwszego elementu

                if (
                    len(val) > 0
                    and isinstance(val[0], dict)
                    and current_depth < max_depth
                ):
                    print_json_tree(
                        val[0],
                        indent + ("    " if last_item else "│   "),
                        last_item,
                        max_depth,
                        current_depth + 1,
                    )
            else:
                val_str = str(val).replace("\n", " ")
                if len(val_str) > 25:
                    val_str = val_str[:25] + "..."
                print(f"{indent}{marker}{key} ({val_type}): {val_str}")


# Ustawienia estetyczene wykresów
sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (10, 5)

data = []
file_path = "github_pr_issue_traces_raw_2025_10k.jsonl"


# Odczyt linia po linii z obsługą kodowania UTF-8
with open(file_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if line:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ Ominięto uszkodzony wiersz {line_num}: {e}")

if data:
    sample_record = data[0]
# Zamiana listy słowników na DataFrame
df_raw = pd.DataFrame(data)


print(f"✅ Pomyślnie wczytano {len(df_raw)} rekordów z pliku JSONL.")
print("Dostępne kolumny:", df_raw.columns.tolist())


# print(" STRUKTURA KLUCZY ZBIORU DATASETU (JSON SCHEMA TREE)")
# print_json_tree(sample_record, max_depth=3)

# 2 Ekstrakcja i Higiena Danych

MAX_LINES_LENGTH_PER_FILE = 150
MAX_TOTAL_LINES = 800

LABEL_KEYWORDS = [
    "bug",
    "fix",
    "hotfix",
    "security",
    "feature",
    "regression",
    "defect",
    "enhanc",
    "doc",
    "chore",
    "refactor",
    "wip",
    "breaking",
    "core",
    "storage",
    "cleanup",
    "style",
    "ci",
    "build",
    "dep",
    "ui",
    "ux",
    "server",
    "tools",
    "test",
]

EXCLUDED_LABEL_KEYWORDS = [
    "p0",
    "p1",
    "p2",
    "p3",
    "p4",
    "priority",
    "prio",
    "urgent",
    "blocker",
    "approved",
    "ci-approved",
    "qa-passed",
    "qa-approved",
    "ready-to-merge",
    "ready-for-merge",
    "needs-review",
    "needs-work",
    "changes-requested",
    "review-requested",
    "merged",
    "closed",
    "sprint",
    "team",
    "jira",
    "epic",
    "squad",
    "dept",
]

EXCLUDED_EXTENSIONS = [
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "composer.lock",
    "cargo.lock",
    "go.sum",
    "pipfile.lock",
    "poetry.lock",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".pdf",
    ".zip",
    ".tar.gz",
    ".ico",
    ".min.js",
    ".min.css",
    ".map",
    ".bundle.js",
]

EXCLUDED_PATHS = [
    "dist/",
    "build/",
    "node_modules/",
    ".next/",
    "vendor/",
    "out/",
]


records = []

labels_counts = Counter()

for idx, row in df_raw.iterrows():
    pr = row.get("pr") if isinstance(row.get("pr"), dict) else {}
    files = row.get("files") if isinstance(row.get("files"), list) else []
    user = pr.get("user") if isinstance(pr.get("user"), dict) else {}
    candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
    username = user.get("login", "") if user else ""
    user_type = user.get("type", "") if user else ""

    commits_count = pr.get("commits") if isinstance(pr.get("commits"), int) else 0

    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    repo_info = base.get("repo") if isinstance(base.get("repo"), dict) else {}
    repo_name = repo_info.get("full_name", "unknown")

    raw_labels_list = [l["name"] for l in pr.get("labels", []) if "name" in l]

    # for label in raw_labels_list:
    #     labels_counts[label.strip()] += 1

    labels_list = []

    for label in raw_labels_list:
        if any(ex_keyword in label.lower() for ex_keyword in EXCLUDED_LABEL_KEYWORDS):
            continue

        if any(keyword in label.lower() for keyword in LABEL_KEYWORDS):
            labels_list.append(label)

    file_paths = []

    for file in files:
        if isinstance(file, dict) and "filename" in file:
            file_paths.append(file["filename"])

    has_tests = any(
        "test" in file["filename"].lower() or "spec" in file["filename"].lower()
        for file in files
    )

    directories_count = len(file_paths)

    # Flagi jakościowe
    api_errors = row.get("api_errors")
    has_api_error = bool(api_errors) if isinstance(api_errors, list) else False
    is_bot = (
        username.endswith("[bot]") or "bot" in username.lower() or user_type == "Bot"
    )

    body = pr.get("body") or ""
    is_empty_body = len(str(body).strip()) < 10

    pr_title = candidate.get("pr_title", "") if candidate else {}
    pr_body = candidate.get("pr_body", "") if candidate else {}

    reviews = row.get("reviews") or []
    review_comments = row.get("review_comments") or []
    has_no_reviews = (len(reviews) == 0) and (len(review_comments) == 0)

    comments_count = len(row.get("pr_comments", []))

    raw_diff = row.get("full_diff")
    if isinstance(raw_diff, str):
        diff_text = raw_diff
    else:
        diff_text = ""

    diff_lines = len(diff_text.splitlines()) if diff_text else 0

    files_diffs = re.split(r"(?=diff --git)", diff_text)

    cleaned_file_blocks = []
    total_lines_count = 0

    for file_block in files_diffs:
        if file_block.strip():
            first_line = file_block.split("\n")[0]
            file_path_match = re.search(r"b/(.+)$", first_line)
            file_path = file_path_match.group(1) if file_path_match else ""
            file_path_lower = file_path.lower()

            is_excluded_file = any(
                file_path_lower.endswith(ext) for ext in EXCLUDED_EXTENSIONS
            )

            is_excluded_dir = any(p_dir in file_path_lower for p_dir in EXCLUDED_PATHS)

            if is_excluded_file or is_excluded_dir:
                cleaned_file_blocks.append(
                    f"[Generated file skipped/lockfile: {file_path}]\n"
                )
            else:
                lines = file_block.split("\n")
                if len(lines) > MAX_LINES_LENGTH_PER_FILE:
                    truncated_lines = lines[:MAX_LINES_LENGTH_PER_FILE]
                    truncated_lines.append(
                        f"\n... [Trimmed diff of file {file_path} - skipping {len(lines) - MAX_LINES_LENGTH_PER_FILE} lines] ..."
                    )
                    file_block = "\n".join(truncated_lines)

                block_line_count = len(file_block.split("\n"))
                if total_lines_count + block_line_count > MAX_TOTAL_LINES:
                    remaining_budget = MAX_TOTAL_LINES - total_lines_count
                    if remaining_budget > 20:
                        truncated_lines = file_block.split("\n")[:remaining_budget]
                        truncated_lines.append(
                            "\n... [Diff line limit reached for entire PR - remaining files trimmed] ..."
                        )
                        cleaned_file_blocks.append("\n".join(truncated_lines))
                    break

                cleaned_file_blocks.append(file_block)
                total_lines_count += block_line_count

        full_diff = "\n".join(cleaned_file_blocks).strip()

    records.append(
        {
            "repo": repo_name,
            "username": username,
            "has_api_error": has_api_error,
            "is_bot": is_bot,
            "is_empty_body": is_empty_body,
            "has_no_reviews": has_no_reviews,
            "state": pr.get("state"),
            "merged": pr.get("merged", False),
            "created_at": pr.get("created_at"),
            "closed_at": pr.get("closed_at"),
            "additions": pr.get("additions", 0) or 0,
            "deletions": pr.get("deletions", 0) or 0,
            "changed_files": pr.get("changed_files", 0) or 0,
            "file_paths_list": ", ".join(file_paths),
            "directories_count": directories_count,
            "diff_lines": diff_lines,
            "full_diff": full_diff,
            "commits_count": commits_count,
            "has_tests": has_tests,
            "labels": ", ".join(labels_list),
            "pr_title": pr_title,
            "pr_body": pr_body,
            "comments_count": comments_count,
        }
    )

clean_df = pd.DataFrame(records)


# Filtrowanie
total_count = len(clean_df)
no_errors = len(clean_df[~clean_df["has_api_error"]])
no_bots = len(clean_df[(~clean_df["has_api_error"]) & (~clean_df["is_bot"])])
final_valid = len(
    clean_df[
        (~clean_df["has_api_error"])
        & (~clean_df["is_bot"])
        & (~clean_df["is_empty_body"])
    ]
)

funnel = pd.DataFrame(
    [
        {"Etap": "1. Całkowita liczba PR", "Liczba": total_count, "Udział %": 100.0},
        {
            "Etap": "2. Bez błędów API",
            "Liczba": no_errors,
            "Udział %": round(no_errors / total_count * 100, 1),
        },
        {
            "Etap": "3. Bez botów (Dependabot itp.)",
            "Liczba": no_bots,
            "Udział %": round(no_bots / total_count * 100, 1),
        },
        {
            "Etap": "4. Czysty zbiór roboczy (z opisem)",
            "Liczba": final_valid,
            "Udział %": round(final_valid / total_count * 100, 1),
        },
    ]
)

print("\n--- FILTROWANIE DANYCH ---")
print(funnel.to_string(index=False))

# Filtracja zbioru roboczego do dalszych statystyk
valid_sample = clean_df[
    (~clean_df["has_api_error"]) & (~clean_df["is_bot"]) & (~clean_df["is_empty_body"])
].copy()


# Analiza Rozmiarów Kodów (LLM Truncation)

print("\n--- PERCENTYLE ROZMIARU ZMIAN KODU (Zbiór Roboczy) ---")
percentiles = valid_sample[
    ["additions", "deletions", "changed_files", "diff_lines"]
].quantile([0.5, 0.75, 0.90, 0.95, 0.99])
print(percentiles.round(1))

# Ground Truth (Statusy i Czasy)
valid_sample["created_dt"] = pd.to_datetime(valid_sample["created_at"], errors="coerce")

valid_sample["closed_dt"] = pd.to_datetime(valid_sample["closed_at"], errors="coerce")

valid_sample["duration_hours"] = (
    valid_sample["closed_dt"] - valid_sample["created_dt"]
).dt.total_seconds() / 3600.0

valid_sample.to_csv("cleaned_dataset.csv", index=True)

# print(labels_counts)

# merged_stats = valid_sample['merged'].value_counts(normalize=True) * 100

# print("\n--- STATUS PR-ów ---")
# print(f"Scalone (Merged = True):  {merged_stats.get(True, 0):.1f}%")
# print(f"Odrzucone (Merged = False): {merged_stats.get(False, 0):.1f}%")

# print(f"\nMediana czasu trwania PR: {valid_sample['duration_hours'].median():.1f} godzin")
# print(f"75. percentyl czasu trwania PR: {valid_sample['duration_hours'].quantile(0.75):.1f} godzin")


# #Różnorodność Repozytoriów

# top_repos = valid_sample['repo'].value_counts().head(10)

# print("\n--- TOP 10 REPOZYTORIÓW W ZBIORZE ---")
# print(top_repos)
