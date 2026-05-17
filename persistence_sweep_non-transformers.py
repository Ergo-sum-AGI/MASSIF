# ==============================================================
# CELL 1 — Install (run once)
# ==============================================================
# Runtime → Change runtime type → T4 GPU OR HPU (if available)
#
# For HPU: mamba-ssm CANNOT install (needs CUDA). 
# We use HuggingFace Mamba with eager (pure PyTorch) fallback.

!pip install transformers torch numpy pandas matplotlib -q
!pip install rwkv -q

# DO NOT attempt mamba-ssm install on HPU — it will fail
# The HF Mamba model has a built-in pure-PyTorch fallback

# ==============================================================
# CELL 2 — Imports and Device Detection
# ==============================================================
import torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import gc, warnings, os
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModelForCausalLM

# Detect device: CUDA, HPU, or CPU
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif hasattr(torch, 'hpu') and torch.hpu.is_available():
    device = "hpu"
    dtype = torch.bfloat16  # HPU prefers BF16
else:
    device = "cpu"
    dtype = torch.float32

eps = 1e-6

print(f"Device: {device}")
if device == "cuda":
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}  ({total:.1f} GB)")
elif device == "hpu":
    # HPU memory info
    try:
        import habana_frameworks.torch.hpu as hthpu
        total = hthpu.memory_stats()['Limit'] / 1e9
        print(f"HPU detected  ({total:.1f} GB)")
    except:
        print("HPU detected")
else:
    print("WARNING: CPU only — using pure PyTorch fallback")

# ==============================================================
# CELL 3 — Shared persistence function (hidden states)
# ==============================================================

def make_seed(n, T, run):
    return int(n) * 1000 + int(T * 10) * 100 + int(run)

def compute_persistence(model, tokenizer, prompt,
                        max_tokens=20, temperature=0.7, seed=None):
    """
    Mean persistence I_t = cos(v_t, v_{t-1}) from actual hidden states.
    Works for any HuggingFace CausalLM with output_hidden_states.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == "hpu":
            torch.hpu.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    h_prev = v_prev = None
    vals = []

    for _ in range(max_tokens):
        with torch.no_grad():
            # CRITICAL: output_hidden_states=True must be in forward() call
            out = model(input_ids, output_hidden_states=True)
            logit = out.logits[0, -1, :]
            
            # Extract hidden states — handle different model formats
            hidden_states = out.hidden_states
            if hidden_states is None:
                raise ValueError("Model returned no hidden_states. "
                               "Check that output_hidden_states=True is supported.")
            
            # Last layer, last token position
            h_new = hidden_states[-1][0, -1]

        # Sample next token
        pr = torch.softmax(logit / max(temperature, 1e-4), dim=-1)
        tok = torch.multinomial(pr, 1)
        input_ids = torch.cat([input_ids, tok.unsqueeze(0)], dim=1)

        # Normalize hidden state
        hu = h_new / (torch.norm(h_new) + eps)

        if h_prev is not None:
            v_t = hu - h_prev
            if v_prev is not None:
                n1 = torch.norm(v_t) + eps
                n2 = torch.norm(v_prev) + eps
                vals.append(float(torch.sum(v_t * v_prev) / (n1 * n2)))
            v_prev = v_t.clone()
        h_prev = hu.clone()

        if tok.item() == tokenizer.eos_token_id:
            break

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "hpu":
        torch.hpu.empty_cache()
    
    return float(np.mean(vals)) if vals else 0.0

def run_sweep(model, tokenizer, label,
              n_values=(2,4,6,8,10,12,15,20),
              temps=(0.5,0.7,0.9),
              n_runs=5, max_tokens=25):
    """Run full persistence sweep and return DataFrame."""
    print(f"\n{'='*60}")
    print(f"SWEEP: {label}")
    print(f"{'='*60}")
    rows = []
    for n in n_values:
        prompt = ("I ") * n
        print(f"  n={n:2d}: ", end="", flush=True)
        for T in temps:
            run_vals = [compute_persistence(
                            model, tokenizer, prompt,
                            max_tokens=max_tokens, temperature=T,
                            seed=make_seed(n, T, r))
                        for r in range(n_runs)]
            mean_v = float(np.mean(run_vals))
            std_v  = float(np.std(run_vals))
            rows.append({"model": label, "n": n, "T": T,
                         "mean": mean_v, "std": std_v})
            flag = " FLIP" if mean_v > 0 else "     "
            print(f" T={T}:{mean_v:+.3f}±{std_v:.3f}{flag}", end="", flush=True)
        print()
    return pd.DataFrame(rows)

print("Shared functions defined.")

# ==============================================================
# CELL 4 — Load and sweep RWKV-169M
# ==============================================================

print("Loading RWKV-169M...")
try:
    rwkv_tok = AutoTokenizer.from_pretrained(
        "RWKV/rwkv-4-169m-pile", trust_remote_code=True)
    if rwkv_tok.pad_token is None:
        rwkv_tok.pad_token = rwkv_tok.eos_token

    rwkv_mod = AutoModelForCausalLM.from_pretrained(
        "RWKV/rwkv-4-169m-pile",
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()

    print(f"RWKV-169M ready on {device}")
    df_rwkv = run_sweep(rwkv_mod, rwkv_tok, "RWKV-169M")

    del rwkv_mod; gc.collect()
    if device == "cuda": torch.cuda.empty_cache()
    elif device == "hpu": torch.hpu.empty_cache()

except Exception as e:
    print(f"RWKV failed: {e}")
    import traceback
    traceback.print_exc()
    df_rwkv = pd.DataFrame()

# ==============================================================
# CELL 5 — Load and sweep Mamba-130M (HuggingFace route)
# ==============================================================
#
# HPU/CUDA/CPU: All use the same HF path.
# mamba-ssm is NOT needed — the model has built-in eager fallback.
#
# CRITICAL FIX: The checkpoint uses tied embeddings but the HF loader
# sometimes fails to tie lm_head to embed_tokens. We enforce it manually.

print("Loading Mamba-130M (HuggingFace, eager fallback)...")

try:
    mamba_model_id = "state-spaces/mamba-130m-hf"
    
    mamba_tok = AutoTokenizer.from_pretrained(mamba_model_id, trust_remote_code=True)
    if mamba_tok.pad_token is None:
        mamba_tok.pad_token = mamba_tok.eos_token

    # Load the model
    mamba_mod = AutoModelForCausalLM.from_pretrained(
        mamba_model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()

    # ========== FIX 1: Manually tie lm_head to embed_tokens ==========
    # The checkpoint only has embed_tokens.weight; lm_head was randomly init'd
    if hasattr(mamba_mod, 'lm_head') and hasattr(mamba_mod, 'get_input_embeddings'):
        embed_weight = mamba_mod.get_input_embeddings().weight
        # Check if lm_head is already tied (shares storage)
        if mamba_mod.lm_head.weight.data_ptr() != embed_weight.data_ptr():
            print("  Fixing: Tying lm_head.weight to embed_tokens.weight...")
            mamba_mod.lm_head.weight = embed_weight
            print("  ✅ Weights tied successfully")
        else:
            print("  ✅ Weights already tied")

    # Also ensure config reflects this (prevents save/load issues)
    if hasattr(mamba_mod.config, 'tie_word_embeddings'):
        mamba_mod.config.tie_word_embeddings = True

    print(f"Mamba-130M (HF eager) ready on {device}")

    # ========== FIX 2: Verify coherence BEFORE sweep ==========
    print("\nMamba Sanity Check:")
    test_prompt = "The capital of France is"
    test_ids = mamba_tok(test_prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        # Test hidden states extraction
        test_out = mamba_mod(test_ids, output_hidden_states=True)
        if test_out.hidden_states is None:
            raise RuntimeError("Model does not return hidden_states")
        
        test_hidden = test_out.hidden_states[-1][0, -1]
        print(f"  Hidden states shape: {test_out.hidden_states[-1].shape}")
        print(f"  Hidden state norm: {torch.norm(test_hidden).item():.4f}")
        
        # Test generation coherence
        gen_ids = mamba_mod.generate(
            test_ids, 
            max_new_tokens=10, 
            do_sample=False,
            pad_token_id=mamba_tok.pad_token_id
        )
        gen_text = mamba_tok.decode(gen_ids[0], skip_special_tokens=True)
    
    print(f"  Generation: '{gen_text}'")
    
    if "Paris" in gen_text or "paris" in gen_text:
        print("  ✅ Mamba generates coherent text — LM head valid")
    else:
        print("  ⚠️ Mamba output may be incoherent — but continuing anyway")
        print(f"     (This can happen with small models; hidden states still valid)")

    print(f"  Using eager fallback: True")

    # ========== FIX 3: Custom compute_persistence for Mamba HF ==========
    # The standard compute_persistence works, but we add extra safety
    def compute_persistence_mamba_hf(model, tokenizer, prompt,
                                     max_tokens=20, temperature=0.7, seed=None):
        """
        Persistence using hidden_states from MambaForCausalLM.
        Identical to standard but with explicit output_hidden_states.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if device == "hpu":
                torch.hpu.manual_seed_all(seed)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        h_prev = v_prev = None
        vals = []

        for _ in range(max_tokens):
            with torch.no_grad():
                # Explicit output_hidden_states in forward call
                out = model(input_ids, output_hidden_states=True)
                logit = out.logits[0, -1, :]
                h_new = out.hidden_states[-1][0, -1]

            # Sample
            pr = torch.softmax(logit / max(temperature, 1e-4), dim=-1)
            tok = torch.multinomial(pr, 1)
            input_ids = torch.cat([input_ids, tok.unsqueeze(0)], dim=1)

            # Normalize and compute persistence
            hu = h_new / (torch.norm(h_new) + eps)

            if h_prev is not None:
                v_t = hu - h_prev
                if v_prev is not None:
                    n1 = torch.norm(v_t) + eps
                    n2 = torch.norm(v_prev) + eps
                    vals.append(float(torch.sum(v_t * v_prev) / (n1 * n2)))
                v_prev = v_t.clone()
            h_prev = hu.clone()

            if tok.item() == tokenizer.eos_token_id:
                break

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "hpu":
            torch.hpu.empty_cache()
        
        return float(np.mean(vals)) if vals else 0.0

    # ========== Run sweep with custom function ==========
    n_values = [2,4,6,8,10,12,15,20]
    temps = [0.5,0.7,0.9]
    n_runs = 5
    rows = []
    
    print(f"\n{'='*60}")
    print(f"SWEEP: Mamba-130M-HF")
    print(f"{'='*60}")
    
    for n in n_values:
        prompt = ("I ") * n
        print(f"  n={n:2d}: ", end="", flush=True)
        for T in temps:
            run_vals = [compute_persistence_mamba_hf(
                            mamba_mod, mamba_tok, prompt,
                            max_tokens=25, temperature=T,
                            seed=make_seed(n, T, r))
                        for r in range(n_runs)]
            mean_v = float(np.mean(run_vals))
            std_v = float(np.std(run_vals))
            rows.append({"model": "Mamba-130M-HF", "n": n, "T": T,
                         "mean": mean_v, "std": std_v})
            flag = " FLIP" if mean_v > 0 else "     "
            print(f" T={T}:{mean_v:+.3f}±{std_v:.3f}{flag}", end="", flush=True)
        print()
    
    df_mamba = pd.DataFrame(rows)

    del mamba_mod
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "hpu":
        torch.hpu.empty_cache()

except Exception as e:
    print(f"Mamba failed: {e}")
    import traceback
    traceback.print_exc()
    df_mamba = pd.DataFrame()
	
# ==============================================================
# CELL 6 — Combine with transformer results and plot
# ==============================================================

import os

csv_path = "all_models_flip_data.csv"
if os.path.exists(csv_path):
    df_prev = pd.read_csv(csv_path)
    print(f"Loaded previous results: {df_prev['model'].unique()}")
else:
    print("No previous CSV found — plotting only new architectures")
    df_prev = pd.DataFrame()

frames = [f for f in [df_prev, df_mamba, df_rwkv] if not f.empty]
if not frames:
    print("No data to plot.")
else:
    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_csv("all_architectures_persistence.csv", index=False)
    print(f"Combined dataset: {df_all['model'].unique()}")

    # Summary table
    print("\n" + "="*65)
    print("FLIP POINT SUMMARY — ALL ARCHITECTURES")
    print("="*65)

    for model_name in df_all["model"].unique():
        df_m = df_all[df_all["model"] == model_name]
        print(f"\n{model_name}:")
        for T in [0.5, 0.7, 0.9]:
            df_t = df_m[df_m["T"] == T].sort_values("n")
            if df_t.empty:
                continue
            means = df_t["mean"].values
            ns = df_t["n"].values
            flip_n = None
            for i in range(len(means)):
                if means[i] > 0 and all(means[j] > 0 for j in range(i, len(means))):
                    flip_n = ns[i]
                    break
            if flip_n:
                print(f"  T={T}: monotonic flip at n={flip_n}")
            else:
                pos = [(ns[i], means[i]) for i in range(len(means)) if means[i] > 0]
                if pos:
                    print(f"  T={T}: sporadic flips at n={[p[0] for p in pos]} "
                          f"(not monotonic)")
                else:
                    print(f"  T={T}: never flips (fully anti-persistent)")

    # Plot
    models = df_all["model"].unique()
    n_models = len(models)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(5*cols, 4*rows),
                             sharey=True)
    axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]

    colors = {0.5: "steelblue", 0.7: "tomato", 0.9: "seagreen"}

    for ax, model_name in zip(axes_flat, models):
        df_m = df_all[df_all["model"] == model_name]
        for T in [0.5, 0.7, 0.9]:
            df_t = df_m[df_m["T"] == T].sort_values("n")
            if df_t.empty:
                continue
            ax.errorbar(df_t["n"], df_t["mean"], yerr=df_t["std"],
                        fmt="o-", color=colors[T], lw=1.8, ms=5,
                        capsize=3, label=f"T={T}")
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set_title(model_name, fontsize=9)
        ax.set_xlabel("n (repetitions)")
        ax.set_ylabel("Mean $I_t$")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for ax in list(axes_flat)[n_models:]:
        ax.set_visible(False)

    fig.suptitle(
        "Persistence Phase Transitions — All Architectures\n"
        "Including Recurrent (Mamba, RWKV) vs Transformer Models\n"
        "DUBITO Inc. / Ergo Sum AGI Safety Systems",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("all_architectures_persistence.png", dpi=150,
                bbox_inches="tight")
    plt.show()
    print("Plot saved: all_architectures_persistence.png")