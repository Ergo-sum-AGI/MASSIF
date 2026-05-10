# =============================================
# MASSIF PHASE 4: Visualisation & Plotting
# =============================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load metrics
df = pd.read_csv(f"{OUTDIR}/metrics_raw_improved.csv")

print("Creating plots...")

# 1. Bar plots - Key Metrics Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['rho_r', 'complexity', 'anisotropy', 'gamma']
titles = ['Phase Coherence ρ_R (higher = more organized)', 
          'Complexity C (higher = more chaotic)', 
          'Anisotropy A', 'Embedding Gap Γ']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[i//2, i%2]
    sns.barplot(data=df, x='category', y=metric, ax=ax, ci=95, palette="viridis")
    ax.set_title(title)
    ax.set_xlabel("Prompt Category")
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/metrics_bar_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Radar Plot (Coherent vs Random vs Harmful vs Safe)
categories_to_plot = ['coherent', 'random', 'harmful', 'safe_benign', 'truthful']
subset = df[df['category'].isin(categories_to_plot)]

# Normalize for radar
metrics_cols = ['rho_r', 'complexity', 'anisotropy']
normalized = subset.groupby('category')[metrics_cols].mean()
normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2*np.pi, len(metrics_cols), endpoint=False).tolist()
angles += angles[:1]

for cat in normalized.index:
    values = normalized.loc[cat].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=cat)
    ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_cols)
ax.set_title("Manifold Geometry Radar Plot")
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.savefig(f"{OUTDIR}/radar_geometry.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Scatter: Complexity vs Phase Coherence
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='complexity', y='rho_r', hue='category', 
                size='num_tokens', sizes=(20, 200), alpha=0.8)
plt.title("Complexity vs Phase Coherence ρ_R (bubble size = num tokens)")
plt.xlabel("Complexity C")
plt.ylabel("Phase Coherence ρ_R")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(f"{OUTDIR}/scatter_complexity_vs_rhor.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Save summary table for preprint/colab
summary_table = df.groupby('category')[['rho_r', 'complexity', 'anisotropy', 'gamma']].mean().round(4)
summary_table.to_csv(f"{OUTDIR}/summary_table.csv")
print("\n=== SUMMARY TABLE ===")
print(summary_table)