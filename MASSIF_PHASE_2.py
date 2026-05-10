# =============================================
# MASSIF PHASE 2: Activation Extraction - FIXED
# =============================================
import pickle
import os
from tqdm import tqdm
import torch
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.decomposition import PCA
import json

# Directories
os.makedirs(OUTDIR, exist_ok=True)
CHECKPOINT_DIR = f"{OUTDIR}/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load prompts
with open(f"{OUTDIR}/prompt_bank_robust.json", "r") as f:
    PROMPT_SETS = json.load(f)

print("Loaded categories:", {k: len(v) for k,v in PROMPT_SETS.items()})

# ========================= CONFIG =========================
MODEL_NAMES = ["gpt2"]                    # Fix gpt2 first before doing medium/large
MAX_PROMPTS_PER_CATEGORY = 25             # Reduced for faster debugging
SEQ_MAX_LENGTH = 80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ====================== FIXED EXTRACTION ======================
results = {}

for model_name in MODEL_NAMES:
    print(f"\n{'='*70}")
    print(f"🚀 EXTRACTING FOR: {model_name.upper()}")
    print(f"{'='*70}")
    
    model = GPT2Model.from_pretrained(
        model_name,
        torch_dtype=torch.float32,           # Use float32 on CPU for stability
        device_map=None
    ).to(DEVICE).eval()
    
    for category, prompts in PROMPT_SETS.items():
        if len(prompts) == 0:
            continue
            
        print(f"   → {category} ({len(prompts)} prompts)")
        activations = []
        selected = prompts[:MAX_PROMPTS_PER_CATEGORY]
        
        for prompt in tqdm(selected, desc=category):
            try:
                # Tokenize
                inputs = tokenizer(prompt, 
                                 return_tensors="pt", 
                                 padding=True, 
                                 truncation=True, 
                                 max_length=SEQ_MAX_LENGTH).to(DEVICE)
                
                # Forward pass - Force hidden states
                with torch.no_grad():
                    outputs = model(**inputs, 
                                  output_hidden_states=True, 
                                  output_attentions=True,
                                  return_dict=True)   # Important
                
                # === FIXED EXTRACTION ===
                hidden_states = outputs.hidden_states
                if hidden_states is None or len(hidden_states) == 0:
                    raise ValueError("No hidden states returned")
                
                last_hidden = hidden_states[-1].cpu().numpy()   # [batch, seq, dim]
                
                # Safe mean pooling
                mean_hidden = last_hidden.mean(axis=1)           # [batch, dim]
                
                # 2D projection
                pca = PCA(n_components=2, random_state=42)
                pos_2d = pca.fit_transform(last_hidden[0])
                
                activations.append({
                    "prompt": prompt[:250],
                    "category": category,
                    "hidden_last": last_hidden,
                    "mean_hidden": mean_hidden,
                    "positions_2d": pos_2d,
                    "num_tokens": last_hidden.shape[1]
                })
                
            except Exception as e:
                print(f"     ⚠️ Skipped: {str(e)[:80]}")
                continue
        
        # Save
        key = f"{model_name}_{category}"
        results[key] = activations
        
        with open(f"{CHECKPOINT_DIR}/partial_{key}.pkl", "wb") as f:
            pickle.dump(activations, f)
    
    # Model checkpoint
    with open(f"{OUTDIR}/activations_{model_name}.pkl", "wb") as f:
        pickle.dump(results, f)
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n🎉 FIXED PHASE 2 COMPLETE!")
print(f"Total keys: {len(results)}")