# =======================================================
# MASSIF PHASE 3: MASSIF Statistical Core (Fixed)
# =======================================================
# MASSIF PHASE 3 v0.3.1 - Updated with fixed trajectories
# =======================================================
with open(f"{OUTDIR}/token_trajectories_fixed.pkl", "rb") as f:
    trajectories = pickle.load(f)
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

print("Loading data for MASSIF Statistical Core v0.3...")

# Load data
with open(f"{OUTDIR}/activations_gpt2.pkl", "rb") as f:
    results = pickle.load(f)

with open(f"{OUTDIR}/token_trajectories.pkl", "rb") as f:
    trajectories = pickle.load(f)

# ====================== METRIC FUNCTIONS (reuse your best) ======================
def compute_embedding_gap(hidden):
    try:
        tokens = hidden[0]
        if tokens.shape[0] < 3:
            return np.nan
        dist_raw = pdist(tokens, 'cosine')
        centered = tokens - tokens.mean(axis=0)
        dist_centered = pdist(centered, 'cosine')
        return np.mean(dist_raw) - np.mean(dist_centered)
    except:
        return np.nan

def compute_phase_coherence(hidden):
    try:
        tokens = hidden[0]
        norms = np.linalg.norm(tokens, axis=1, keepdims=True)
        normalized = tokens / (norms + 1e-8)
        gram = normalized @ normalized.T
        eig = np.linalg.eigvalsh(gram)
        return float(np.max(eig) / (np.sum(np.abs(eig)) + 1e-8))
    except:
        return np.nan

def compute_anisotropy(hidden):
    try:
        tokens = hidden[0].T
        cov = np.cov(tokens)
        eig = np.linalg.eigvalsh(cov)
        return float(np.max(eig) / (np.sum(eig) + 1e-8))
    except:
        return np.nan

def compute_complexity(hidden):
    try:
        tokens = hidden[0]
        norms = np.linalg.norm(tokens, axis=1)
        norms = norms / (norms.sum() + 1e-8)
        return float(stats.entropy(norms))
    except:
        return np.nan

# ====================== PERPLEXITY CONTROL ======================
from transformers import GPT2LMHeadModel, GPT2Tokenizer

lm_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def compute_perplexity(prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = lm_model(**inputs, labels=inputs.input_ids)
        return float(torch.exp(outputs.loss))
    except:
        return np.nan

# ====================== MAIN DATAFRAME ======================
metrics_list = []

for key, acts in results.items():
    model, category = key.split("_", 1)
    for act in acts:
        hidden = act.get("hidden_last")
        prompt = act.get("prompt", "")
        ppl = compute_perplexity(prompt)
        
        row = {
            "model": model,
            "category": category,
            "is_unsafe": 1 if category in ["harmful", "jailbreak", "matched_unsafe", "deceptive"] else 0,
            "gamma": compute_embedding_gap(hidden),
            "rho_r": compute_phase_coherence(hidden),
            "anisotropy": compute_anisotropy(hidden),
            "complexity": compute_complexity(hidden),
            "perplexity": ppl,
            "num_tokens": act.get("num_tokens", 0)
        }
        metrics_list.append(row)

df = pd.DataFrame(metrics_list)

# ====================== BOOTSTRAP CI (Pure NumPy) ======================
def bootstrap_ci(data, n_boot=2000, alpha=0.05):
    data = np.array(data)
    boots = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boots.append(np.mean(sample))
    lower = np.percentile(boots, 100*alpha/2)
    upper = np.percentile(boots, 100*(1-alpha/2))
    return lower, upper

print("\n=== BOOTSTRAP 95% CI for ρ_R ===")
for cat in sorted(df['category'].unique()):
    data = df[df['category'] == cat]['rho_r'].dropna().values
    if len(data) >= 5:
        mean_val = data.mean()
        lower, upper = bootstrap_ci(data)
        print(f"{cat:15} ρ_R = {mean_val:.4f}  [95% CI: {lower:.4f} — {upper:.4f}]")

# ====================== PERMUTATION TEST ======================
print("\n=== PERMUTATION TEST: Coherent vs Harmful (ρ_R) ===")
coh = df[df['category']=='coherent']['rho_r'].dropna().values
harm = df[df['category']=='harmful']['rho_r'].dropna().values

if len(coh) > 0 and len(harm) > 0:
    obs_diff = np.mean(coh) - np.mean(harm)
    perm_diffs = []
    for _ in range(2000):                     # reduced for speed on CPU
        combined = np.concatenate([coh, harm])
        np.random.shuffle(combined)
        perm_diffs.append(np.mean(combined[:len(coh)]) - np.mean(combined[len(coh):]))
    
    p_value = (np.abs(perm_diffs) >= np.abs(obs_diff)).mean()
    print(f"Observed diff (Coherent - Harmful): {obs_diff:.4f}")
    print(f"Permutation p-value: {p_value:.4f} {'***' if p_value < 0.01 else ' **' if p_value < 0.05 else ''}")

# ====================== ROC-AUC CLASSIFIER ======================
print("\n=== GEOMETRIC CLASSIFIER PERFORMANCE (Unsafe vs Safe) ===")
features = ['rho_r', 'complexity', 'anisotropy', 'gamma']
X = df[features].fillna(df[features].median())
y = df['is_unsafe']

auc_scores = {}
for feat in features:
    # Flip sign for features where higher = more unsafe
    scores = -X[feat] if feat in ['complexity', 'anisotropy'] else X[feat]
    auc = roc_auc_score(y, scores)
    auc_scores[feat] = auc

for f, a in auc_scores.items():
    print(f"  {f:12} AUC: {a:.4f}")

# Combined example
combined_score = X['rho_r'] - 0.5*X['complexity']
print(f"Combined (ρ_R - 0.5×Complexity) AUC: {roc_auc_score(y, combined_score):.4f}")

# ====================== EARLY DIVERGENCE ======================
print("\n=== IMPROVED EARLY DIVERGENCE ANALYSIS ===")
for cat in ["coherent", "harmful", "random", "matched_safe", "matched_unsafe"]:
    if cat in trajectories and trajectories[cat]:
        trajs = trajectories[cat]
        early_rhos = []
        for t in trajs:
            traj = t['trajectory']
            if len(traj) >= 6:
                early_rhos.append(traj[5]['rho_r'])   # step 6
        if early_rhos:
            mean_r = np.nanmean(early_rhos)
            std_r = np.nanstd(early_rhos)
            print(f"{cat:15} Early ρ_R (step ~6): {mean_r:.4f} ± {std_r:.4f} (n={len(early_rhos)})")

# ====================== SAVE ======================
df.to_csv(f"{OUTDIR}/massif_metrics_v03.csv", index=False)
print(f"\n✅ MASSIF Statistical Core v0.3 COMPLETE!")
print(f"Results saved to {OUTDIR}/massif_metrics_v03.csv")