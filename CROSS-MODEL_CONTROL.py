# ==============================================================
# CROSS-MODEL CONTROL EXPERIMENT
# Testing whether repetitive "I" flips across multiple architectures
# ==============================================================

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import time

print("=" * 70)
print("CROSS-MODEL CONTROL EXPERIMENT")
print("Testing whether repetitive 'I' flips across architectures")
print("=" * 70)

# ==============================================================
# MODELS TO TEST (all open, no special requirements)
# ==============================================================

MODELS = [
    ("GPT-2 Small", "gpt2"),
    ("GPT-2 Medium", "gpt2-medium"),
    ("GPT-2 Large", "gpt2-large"),
    ("GPT-2 XL", "gpt2-xl"),
    ("TinyLlama 1.1B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ("Qwen2 0.5B", "Qwen/Qwen2-0.5B"),
    ("Bloom-560M", "bigscience/bloom-560m"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# ==============================================================
# PERSISTENCE COMPUTATION (Single run per condition for speed)
# ==============================================================

def compute_persistence(model, tokenizer, prompt, max_tokens=20, temperature=0.7, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs.input_ids
    
    h_prev = None
    v_prev = None
    persistence_vals = []
    
    with torch.no_grad():
        for step in range(max_tokens):
            outputs = model(input_ids, output_hidden_states=True)
            logits = outputs.logits[0, -1, :]
            h_new = outputs.hidden_states[-1][0, -1]
            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(logits, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            hu = h_new / (torch.norm(h_new) + 1e-6)
            
            if h_prev is not None:
                v_t = hu - h_prev
                
                if v_prev is not None:
                    n1 = torch.norm(v_t) + 1e-6
                    n2 = torch.norm(v_prev) + 1e-6
                    persistence = float(torch.sum(v_t * v_prev) / (n1 * n2))
                    persistence_vals.append(persistence)
                v_prev = v_t.clone()
            
            h_prev = hu.clone()
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return np.mean(persistence_vals) if persistence_vals else 0.0


# ==============================================================
# CONTROL PROMPTS (n=8, same as Bloom experiment)
# ==============================================================

n = 8
CONTROL_PROMPTS = {
    "repetitive_I": ("I " * n).strip(),
    "repetitive_the": ("the " * n).strip(),
    "varied_self": "I think I know I feel I believe I doubt I am",
    "varied_neutral": "The cat sat on the mat near the door by the",
}

print(f"Testing at n={n} repetitions\n")
print(f"{'Model':<20} {'repetitive_I':>14} {'repetitive_the':>14} {'varied_self':>14} {'varied_neutral':>14}")
print("-" * 80)

results = []

for model_name, model_id in MODELS:
    print(f"\nLoading {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device).eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        row = {'model': model_name}
        
        for prompt_name, prompt in CONTROL_PROMPTS.items():
            persistence = compute_persistence(model, tokenizer, prompt, max_tokens=20, temperature=0.7)
            row[prompt_name] = persistence
        
        results.append(row)
        
        # Print inline
        print(f"{model_name:<20} {row['repetitive_I']:>+14.4f} {row['repetitive_the']:>+14.4f} "
              f"{row['varied_self']:>+14.4f} {row['varied_neutral']:>+14.4f}")
        
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        continue

# ==============================================================
# RESULTS TABLE
# ==============================================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

df = pd.DataFrame(results)
print(df.to_string(index=False))

# Determine which models show the "Bloom pattern" (only repetitive_I flips)
print("\n" + "=" * 70)
print("PATTERN ANALYSIS")
print("=" * 70)

for _, row in df.iterrows():
    model = row['model']
    i_flip = row['repetitive_I'] > 0
    the_flip = row['repetitive_the'] > 0
    self_flip = row['varied_self'] > 0
    
    if i_flip and not the_flip and not self_flip:
        pattern = "✅ Bloom pattern (only 'I' flips)"
    elif i_flip and the_flip:
        pattern = "⚠️ Repetition pattern (both 'I' and 'the' flip)"
    elif not i_flip:
        pattern = "❌ No flip (anti-persistent across all)"
    else:
        pattern = "❓ Mixed / other"
    
    print(f"{model:<20} {pattern}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

# Count models showing Bloom pattern
bloom_pattern_count = sum(1 for _, row in df.iterrows() 
                          if row['repetitive_I'] > 0 and row['repetitive_the'] <= 0 and row['varied_self'] <= 0)

print(f"\nModels showing 'Bloom pattern' (only repetitive 'I' flips): {bloom_pattern_count}/{len(df)}")

if bloom_pattern_count == len(df):
    print("\n✅ ALL MODELS show the same pattern as Bloom")
    print("   → Token-specific effect is GENERALIZABLE across architectures")
elif bloom_pattern_count > 0:
    print("\n⚠️ SOME MODELS show the Bloom pattern, others do not")
    print("   → Effect is architecture-dependent")
else:
    print("\n❌ NO OTHER MODEL shows the Bloom pattern")
    print("   → Bloom's behavior is IDIOSYNCRATIC, not generalizable")