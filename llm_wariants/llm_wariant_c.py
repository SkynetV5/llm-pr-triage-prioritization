import json
import ollama
import math
import pandas as pd
import time
import re


try:
    from json_repair import repair_json

    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

PRIORITY_MAP = {"P0": 3, "P1": 2, "P2": 1, "P3": 0}


def parse_llm_json(answer_str: str) -> dict:
    """Fix parser from output LLM: json.loads -> json_repair -> regex fallback."""
    if not answer_str or not isinstance(answer_str, str):
        return {"priority": "P2", "reasoning": "Empty answer from LLM"}

    
    try:
        return json.loads(answer_str)
    except Exception:
        pass

   
    if HAS_JSON_REPAIR:
        try:
            repaired = repair_json(answer_str)
            return json.loads(repaired)
        except Exception:
            pass

    match = re.search(r"\b(P[0-3])\b", answer_str, re.IGNORECASE)
    if match:
        p_code = match.group(1).upper()
        return {
            "priority": p_code,
            "reasoning": (
                f"Recovered priority {p_code} from text using Regex (JSON syntax error)"
            ),
        }


    return {
        "priority": "P2",
        "reasoning": "Fatal JSON formatting error - default P2",
    }


def predict_priority(title, description, full_diff):

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
    PR Full Diff = {full_diff}

    Respond strictly in JSON format with two keys:
    1. "priority": exactly one string from ["P0","P1", "P2", "P3"]
    2. "reasoning": a concise 1-2 sentence explanation for your decision
    """

    try:
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}], format="json", options={"temperature": 0.0, "num_ctx": 2048, "num_predict": 256})

        answer_raw = response["message"]["content"]

        answer = parse_llm_json(answer_raw)

        reasoning = str(answer.get("reasoning", ""))

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

valid_sample_gt = pd.read_csv("github_prs_ground_truth_sample_2000.csv" ,index_col=0, parse_dates=["created_at", "closed_at"] )


print(f"Rozpoczynam przetwarzanie {len(valid_sample_gt)} PR-ów za pomocą LLM (Wariant C)")
    


pred_texts = []
pred_class_nums = []
reasonings = []

start_time = time.time()

for idx, (i, row ) in enumerate(valid_sample_gt.iterrows(), start=1):

    result = predict_priority(row["pr_title"], row["pr_body"], row["full_diff"])

    pred_texts.append(result["priority_text"])
    pred_class_nums.append(result["priority_num"]) 
    reasonings.append(result["reasoning"])

    if idx % 25 == 0 or idx == len(valid_sample_gt):
        elapsed = time.time() - start_time
        avg_per_pr = elapsed / idx
        print( f"Progress: {idx}/{len(valid_sample_gt)} PR-ów | Czas: {elapsed:.1f}s | Średnio:" f" {avg_per_pr:.2f}s/PR" )


valid_sample_gt["llm_pred_priority_text"] = pred_texts
valid_sample_gt["llm_pred_class_num"] = pred_class_nums
valid_sample_gt["llm_reasoning"] = reasonings

output_filename = "github_prs_llm_wariant_c.csv"

valid_sample_gt.to_csv( output_filename, index=True)

total_time = time.time() - start_time 
print( f"\\n ZAKOŃCZONO! Zapisano wyniki do pliku: {output_filename}" f" (Łączny czas: {total_time:.2f}s)" )
