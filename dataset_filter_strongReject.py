import pandas as pd
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset, evaluate
import os

os.environ["HF_TOKEN"] = ""

# 1. Load the CSV file
csv_path = "/train_dataset/all_evaluated_00000_59999.csv"
df = pd.read_csv(csv_path)

# 2. Extract and rename the required columns to map them to the expected schema
# The evaluator expects 'forbidden_prompt' and 'response'.
df_subset = df[['adversarial', 'target_response']].copy()
df_subset = df_subset.rename(columns={
    'adversarial': 'forbidden_prompt',
    'target_response': 'response'
})

hf_dataset = Dataset.from_pandas(df_subset)

# 3. Evaluate the dataset using 'strongreject_finetuned'
EVALUATOR = "strongreject_finetuned"
evaluated_dataset = evaluate_dataset(hf_dataset, evaluators=[EVALUATOR])

# 4. Convert back to pandas to easily inspect and filter the result
result_df = evaluated_dataset.to_pandas()

# Filter the dataset: keep only scores >= 0.5 (as per standard thresholding, and user request to save samples with >0.5)
score_col = [col for col in result_df.columns if 'score' in col.lower()]
active_score_col = score_col[0] if score_col else 'score'

high_scores_df = result_df[result_df[active_score_col] >= 0.5]

# Save the filtered high score samples to a new CSV file
output_csv_path = "/train_dataset/evaluated_high_scores_05higher.csv"
high_scores_df.to_csv(output_csv_path, index=False)

print(f"Evaluation and filtering completed successfully!")
print(f"Saved {len(high_scores_df)} samples with score >= 0.5 to '{output_csv_path}'")