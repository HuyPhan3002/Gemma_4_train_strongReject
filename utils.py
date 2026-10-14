import os
# Prefer the caller's token; do not replace it with a source-controlled value.
os.environ.setdefault("HF_TOKEN", "")
os.environ.pop("SSL_CERT_FILE", None)  
os.environ.pop("REQUESTS_CA_BUNDLE", None) 
os.environ.pop("CURL_CA_BUNDLE", None)  
import torch
from transformers import AutoTokenizer, AutoModel

repo_name = "nvidia/Nemotron-Labs-Diffusion-3B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_name, trust_remote_code=True)
model = model.to(device=device, dtype=torch.float16)
model.eval()

def get_response(prompt: str) -> str:
    """Generate a response from the model (Nemotron 3B) based on the given prompt."""
    history = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    out_ids, nfe = model.ar_generate(prompt_ids, max_new_tokens=512)
    reply = tokenizer.decode(out_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    return reply

def extract_scores(results: list[dict[str, float]], key: str = "score") -> torch.Tensor:
    """Extract float values for a specific metric key across all evaluators.
    
    Args:
        results: Output from StrongREJECT evaluate().
        key: The dictionary key to extract (defaults to "score").
        
    Returns:
        float score corresponding to each evaluator (mean of float scores list if evaluate function returns multiple).
    """
    scores = [res[key] for res in results if key in res and isinstance(res[key], (float, int))]
    return torch.tensor(sum(scores) / len(scores) if scores else 0.0)
