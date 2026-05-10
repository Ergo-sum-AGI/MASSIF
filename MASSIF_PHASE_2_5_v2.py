# =============================================----
# MASSIF PHASE 2.5 FIXED:
# Token-wise Trajectory Geometry (Early Divergence)
# =================================================
# =============================================
# MASSIF PHASE 2.5 v2: Robust Token-wise Trajectories
# =============================================
import torch
import numpy as np
from tqdm import tqdm

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()

def safe_phase_coherence(hidden):
    """Robust version that works on single vectors"""
    try:
        if hidden.ndim == 1:
            hidden = hidden.reshape(1, -1)
        if hidden.shape[0] == 1:
            return 1.0  # single vector = perfect coherence by definition
        norms = np.linalg.norm(hidden, axis=1, keepdims=True)
        normalized = hidden / (norms + 1e-8)
        gram = normalized @ normalized.T
        eig = np.linalg.eigvalsh(gram)
        return float(np.max(eig) / (np.sum(np.abs(eig)) + 1e-8))
    except:
        return np.nan

def safe_anisotropy(hidden):
    try:
        if hidden.ndim == 1:
            hidden = hidden.reshape(1, -1)
        if hidden.shape[0] == 1:
            return 1.0
        cov = np.cov(hidden.T)
        eig = np.linalg.eigvalsh(cov)
        return float(np.max(eig) / (np.sum(eig) + 1e-8))
    except:
        return np.nan

def safe_complexity(hidden):
    try:
        if hidden.ndim == 1:
            hidden = hidden.reshape(1, -1)
        if hidden.shape[0] == 1:
            return 0.0
        norms = np.linalg.norm(hidden, axis=1)
        norms = norms / (norms.sum() + 1e-8)
        return float(stats.entropy(norms))
    except:
        return np.nan

def extract_token_trajectory(prompt, max_steps=40):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    trajectory = []
    
    with torch.no_grad():
        for i in range(1, min(max_steps + 1, inputs.input_ids.shape[1] + 8)):
            partial = {k: v[:, :i] for k, v in inputs.items()}
            outputs = model(**partial, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
            
            traj_point = {
                "step": i,
                "token": tokenizer.decode([inputs.input_ids[0, i-1]]).strip(),
                "rho_r": safe_phase_coherence(last_hidden),
                "anisotropy": safe_anisotropy(last_hidden),
                "complexity": safe_complexity(last_hidden),
                "hidden_norm": float(np.linalg.norm(last_hidden))
            }
            trajectory.append(traj_point)
    return trajectory

# Re-run extraction
trajectory_results = {}
key_cats = ["coherent", "harmful", "safe_benign", "random"]

for category in key_cats:
    if category not in PROMPT_SETS or not PROMPT_SETS[category]:
        continue
    print(f"Extracting trajectories for {category}...")
    traj_list = []
    for prompt in tqdm(PROMPT_SETS[category][:15]):
        try:
            traj = extract_token_trajectory(prompt)
            if traj and len(traj) > 5:
                traj_list.append({"prompt": prompt[:150], "category": category, "trajectory": traj})
        except:
            continue
    trajectory_results[category] = traj_list

with open(f"{OUTDIR}/token_trajectories_v2.pkl", "wb") as f:
    pickle.dump(trajectory_results, f)

print("✅ Robust token-wise trajectories saved!")