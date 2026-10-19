import json
import ollama
import math
import pandas as pd


PRIORITY_MAP = {"P0": 3, "P1": 2, "P2": 1, "P3": 0}


def predict_priority(title, description):

    if not isinstance(title, str) or not str(title).strip(): 
        title = "Not provided title" 
    else: title = title[:300]

    if not isinstance(description, str) or not str(description).strip():
        description = "Not provided description"
    else: description = description[:2500]


    prompt = f"""You are an expert lead software engineer performing Code Review Triage.
    Analyze the following Pull Request (PR) and assign it to one of 4 priority levels:

    - P0 (Critical): Emergency hotfixes, production outages, severe security vulnerabilities, build breakers. 
    - P1 (High): Major bug fixes, core functional improvements, urgent feature work. 
    - P2 (Medium): Standard features, routine bug fixes, general refactoring, non-blocker maintenance. 
    - P3 (Low): Minor typos, documentation updates, code formatting, trivial cleanup.

    PR Title = {title}
    PR Description = {description}

    Respond strictly in JSON format with two keys:
    1. "priority": exactly one string from ["P0","P1", "P2", "P3"]
    2. "reasoning": a concise 1-2 sentence explanation for your decision
    """

    try:
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}], format="json", options={"temperature": 0.0, "num_ctx": 2048,})

        answer = response["message"]["content"]

        answer = json.loads(answer)

        priority = answer.get("priority", "P2").strip().upper()

        priority_num = PRIORITY_MAP.get(priority,1)

        reasoning = answer.get("reasoning", "")

        return {
            "priority_text": priority,
            "priority_num": priority_num,
            "reasoning": reasoning,
        }

    except Exception as e:
         print(f"Błąd Ollamy: {e}")
         return {
                    "priority_text": 'P2',
                    "priority_num": 1,
                    "reasoning": str(e),
                }

valid_sample_gt = pd.read_csv("github_prs_ground_truth.csv" ,index_col=0, parse_dates=["created_at", "closed_at"] )


#test for first row
first_row = valid_sample_gt.iloc[0]
print(first_row)
title = first_row["pr_title"]
desc = first_row["pr_body"]
gt_value = first_row["gt_class_num"]

result = predict_priority(title,desc)

print("=== TEST NA 1 PR-ZE Z BAZY DANYCH ===") 
print(f"Tytuł PR: {title}") 
print(f"Prawdziwy GT: {gt_value}") 
print( f"Predykcja Ollamy: {result['priority_text']}" f" (numerycznie: {result['priority_num']})" ) 
print(f"Uzasadnienie LLM: {result['reasoning']}")

    
