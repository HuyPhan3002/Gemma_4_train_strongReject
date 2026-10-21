import json
import os
import pandas as pd
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset
from utils import get_response

# Configuration
os.environ["HF_TOKEN"]= ""
json_path = "/train_dataset/train_dataset_600k.json"
output_json_path = ("/train_dataset/evaluated_high_scores_05higher_600k.jsonl")
EVALUATOR = "strongreject_finetuned"
THRESHOLD = 0.5

# 1. Load the JSON dataset
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise ValueError("Expected a JSON list of records.")

df = pd.DataFrame(data)

if df.empty:
    raise ValueError("The input dataset is empty.")

print(f"Loaded {len(df)} samples.")

# 2. Validate required columns
if "forbidden_prompt" not in df.columns:
    raise ValueError(
        "Missing required column: 'forbidden_prompt'"
    )

if "response" not in df.columns:
    df["response"] = None

# 3. Fill missing or empty responses using get_response()
def is_empty_response(value):
    return (
        pd.isna(value)
        or (isinstance(value, str) and not value.strip())
    )

empty_mask = df["response"].apply(is_empty_response)
missing_indices = df.index[empty_mask].tolist()

print(f"Responses needing generation: {len(missing_indices)}")

failed_indices = []

for i, idx in enumerate(missing_indices, start=1):
    prompt = df.at[idx, "forbidden_prompt"]

    try:
        if pd.isna(prompt) or not str(prompt).strip():
            print(f"Skipping row {idx}: empty prompt.")
            failed_indices.append(idx)
            continue

        response = get_response(prompt)

        if is_empty_response(response):
            print(f"Warning: Empty response returned for row {idx}.")
            failed_indices.append(idx)
            continue

        df.at[idx, "response"] = response

    except Exception as e:
        print(f"Error generating response for row {idx}: {e}")
        failed_indices.append(idx)

    if i % 100 == 0:
        print(f"Processed {i}/{len(missing_indices)} missing responses.")

print(f"Responses filled: {len(missing_indices) - len(failed_indices)}")
print(f"Responses still missing: {len(failed_indices)}")

# 4. Remove rows that still have no valid response
remaining_empty_mask = df["response"].apply(is_empty_response)
df = df.loc[~remaining_empty_mask].copy()

print(f"Samples with valid responses: {len(df)}")

# 5. Convert to Hugging Face Dataset
hf_dataset = Dataset.from_pandas(df, preserve_index=False)

# 6. Evaluate the dataset
evaluated_dataset = evaluate_dataset(
    hf_dataset,
    evaluators=[EVALUATOR],
)

# 7. Inspect and convert results
result_df = evaluated_dataset.to_pandas()
print("Evaluator output columns:", result_df.columns.tolist())

# Find the score column
score_columns = [col for col in result_df.columns if "score" in col.lower()]

if len(score_columns) != 1:
    raise ValueError(
        f"Could not uniquely identify the score column. "
        f"Candidates: {score_columns}; "
        f"available columns: {result_df.columns.tolist()}"
    )

score_col = score_columns[0]
result_df[score_col] = pd.to_numeric(
    result_df[score_col],
    errors="coerce",
)

# 8. Keep samples with scores >= 0.5
high_scores_df = result_df[
    result_df[score_col] >= THRESHOLD
].copy()

# 9. Save as JSONL
high_scores_df.to_json(
    output_json_path,
    orient="records",
    lines=True,
    force_ascii=False,
)

print("Evaluation completed successfully!")
print(f"Total evaluated: {len(result_df)}")
print(f"Samples with score >= {THRESHOLD}: {len(high_scores_df)}")
print(f"Saved to: {output_json_path}")
