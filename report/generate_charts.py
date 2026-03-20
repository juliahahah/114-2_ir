"""
Generate visualization charts for the Setwise Ranking experiment report.
Outputs: PNG images saved to report/figures/
"""

import os
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Setup ──
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

# ── Data ──
methods =       ["Pointwise", "Pairwise\n(Heapsort)", "Listwise\n(Generation)", "Listwise\n(Likelihood)", "Setwise\n(Direct)", "Setwise\n(Heapsort)", "Setwise\n(BubbleSort)"]
methods_short = ["Pointwise", "Pairwise", "Listwise Gen", "Listwise LL", "Setwise Dir", "Setwise Heap", "Setwise Bubble"]
ndcg =          [0.8510, 0.5204, 0.8756, 0.8498, 0.7415, 0.8777, 0.8717]
latency =       [0.964,  4.342,  1.552,  0.457,  0.464,  1.885,  2.184]
fw_passes =     [10,     32,     6,      1,      1,      10,     13]

# Color scheme: blue=Pointwise, red=Pairwise, purple=Listwise, orange/gold=Setwise
colors = ['#4A90D9', '#C0392B', '#8E44AD', '#9B59B6', '#F39C12', '#E67E22', '#D68910']
paradigm_colors = {
    'Pointwise': '#4A90D9',
    'Pairwise': '#C0392B',
    'Listwise': '#8E44AD',
    'Setwise': '#F39C12',
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Figure 1: NDCG Comparison Bar Chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(methods, ndcg, color=colors, edgecolor='white', linewidth=1.5, width=0.7)

# Add value labels on bars
for bar, val in zip(bars, ndcg):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('NDCG Score')
ax.set_title('Ranking Effectiveness Comparison (NDCG)')
ax.set_ylim(0, 1.05)
ax.axhline(y=0.8777, color='#E67E22', linestyle='--', alpha=0.4, label='Best: Setwise Heapsort')
ax.legend(loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('figures/fig1_ndcg_comparison.png')
plt.close()
print("  ✓ Figure 1: NDCG comparison saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Figure 2: Latency Comparison Bar Chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(methods, latency, color=colors, edgecolor='white', linewidth=1.5, width=0.7)

for bar, val in zip(bars, latency):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Latency (seconds)')
ax.set_title('Computational Efficiency Comparison (Latency)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('figures/fig2_latency_comparison.png')
plt.close()
print("  ✓ Figure 2: Latency comparison saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Figure 3: Forward Passes Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(methods, fw_passes, color=colors, edgecolor='white', linewidth=1.5, width=0.7)

for bar, val in zip(bars, fw_passes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(val), ha='center', va='bottom', fontweight='bold', fontsize=13)

ax.set_ylabel('Number of Forward Passes')
ax.set_title('LLM Inference Count Comparison')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('figures/fig3_forward_passes.png')
plt.close()
print("  ✓ Figure 3: Forward passes saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Figure 4: NDCG vs Latency Scatter Plot (The Missing Quadrant)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(10, 8))

# Plot each method
for i, (m, n, l) in enumerate(zip(methods_short, ndcg, latency)):
    if 'Pointwise' in m:
        c, marker = paradigm_colors['Pointwise'], 'o'
    elif 'Pairwise' in m:
        c, marker = paradigm_colors['Pairwise'], 's'
    elif 'Listwise' in m:
        c, marker = paradigm_colors['Listwise'], '^'
    else:
        c, marker = paradigm_colors['Setwise'], 'D'

    ax.scatter(l, n, c=c, s=200, marker=marker, zorder=5, edgecolors='white', linewidth=1.5)
    ax.annotate(m, (l, n), textcoords="offset points", xytext=(10, 8),
                fontsize=9, fontweight='bold')

# Quadrant lines
mid_x = 2.5
mid_y = 0.75
ax.axvline(x=mid_x, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=mid_y, color='gray', linestyle=':', alpha=0.5)

# Quadrant labels
ax.text(0.3, 0.98, '★ HIGH Quality\nLOW Latency\n(The Target)', transform=ax.transAxes,
        fontsize=10, color='#27ae60', fontweight='bold', va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', alpha=0.8))
ax.text(0.7, 0.15, 'LOW Quality\nHIGH Latency\n(Worst)', transform=ax.transAxes,
        fontsize=10, color='#c0392b', fontweight='bold', va='bottom', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdedec', alpha=0.8))

# Legend
legend_handles = [
    mpatches.Patch(color=paradigm_colors['Pointwise'], label='Pointwise'),
    mpatches.Patch(color=paradigm_colors['Pairwise'], label='Pairwise'),
    mpatches.Patch(color=paradigm_colors['Listwise'], label='Listwise'),
    mpatches.Patch(color=paradigm_colors['Setwise'], label='Setwise'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=11)

ax.set_xlabel('Latency (seconds) →')
ax.set_ylabel('NDCG Score (Effectiveness) →')
ax.set_title('Effectiveness vs. Efficiency: The Missing Quadrant')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('figures/fig4_ndcg_vs_latency.png')
plt.close()
print("  ✓ Figure 4: NDCG vs Latency scatter saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Figure 5: Efficiency Ratio (NDCG per Forward Pass)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
efficiency = [n / f for n, f in zip(ndcg, fw_passes)]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(methods, efficiency, color=colors, edgecolor='white', linewidth=1.5, width=0.7)

for bar, val in zip(bars, efficiency):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_ylabel('NDCG per Forward Pass')
ax.set_title('Efficiency Ratio: Ranking Quality per LLM Inference')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('figures/fig5_efficiency_ratio.png')
plt.close()
print("  ✓ Figure 5: Efficiency ratio saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Figure 6: Radar Chart — Multi-Dimensional Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
categories = ['NDCG', 'Speed\n(1/Latency)', 'Efficiency\n(1/FW Passes)', 'MRR']
N_cats = len(categories)

# Normalize values to 0-1 range
max_latency = max(latency)
max_fw = max(fw_passes)

# Select 4 representative methods
selected = [
    ("Pointwise",         0.8510, 1 - 0.964/max_latency,  1 - 10/max_fw,  1.0, paradigm_colors['Pointwise']),
    ("Pairwise (Heap)",   0.5204, 1 - 4.342/max_latency,  1 - 32/max_fw,  1.0, paradigm_colors['Pairwise']),
    ("Listwise (Gen)",    0.8756, 1 - 1.552/max_latency,  1 - 6/max_fw,   1.0, paradigm_colors['Listwise']),
    ("Setwise (Heap)",    0.8777, 1 - 1.885/max_latency,  1 - 10/max_fw,  1.0, paradigm_colors['Setwise']),
]

angles = np.linspace(0, 2 * np.pi, N_cats, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for name, *vals, color in selected:
    vals_list = list(vals)
    vals_list += vals_list[:1]
    ax.plot(angles, vals_list, 'o-', linewidth=2, label=name, color=color)
    ax.fill(angles, vals_list, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_title('Multi-Dimensional Comparison\n(Normalized Scores)', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.tight_layout()
plt.savefig('figures/fig6_radar_comparison.png')
plt.close()
print("  ✓ Figure 6: Radar chart saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Figure 7: Ranking Results Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
relevance = {0: 3, 1: 2, 2: 1, 3: 0, 4: 2, 5: 0, 6: 1, 7: 3, 8: 0, 9: 1}
rankings = {
    "Pointwise":       [4, 0, 1, 7, 6, 2, 9, 5, 3, 8],
    "Pairwise":        [6, 0, 2, 1, 9],
    "Listwise Gen":    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Listwise LL":     [0, 6, 2, 1, 3, 4, 8, 7, 5, 9],
    "Setwise Direct":  [6, 0, 1, 2, 4, 7, 9, 3, 5, 8],
    "Setwise Heap":    [0, 6, 7, 4, 9],
    "Setwise Bubble":  [0, 6, 7, 9, 4],
    "Ideal":           [0, 7, 1, 4, 2, 6, 9, 3, 5, 8],
}

fig, ax = plt.subplots(figsize=(14, 6))

method_names = list(rankings.keys())
max_len = max(len(v) for v in rankings.values())

# Build heatmap data
heatmap_data = np.full((len(method_names), max_len), -1.0)
for i, (name, rank_list) in enumerate(rankings.items()):
    for j, doc_idx in enumerate(rank_list):
        heatmap_data[i, j] = relevance[doc_idx]

# Custom colormap: 0=red, 1=yellow, 2=lightgreen, 3=darkgreen, -1=gray
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap_colors = ['#f0f0f0', '#e74c3c', '#f1c40f', '#2ecc71', '#27ae60']
cmap = ListedColormap(cmap_colors)
bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, cmap.N)

im = ax.imshow(heatmap_data, cmap=cmap, norm=norm, aspect='auto')

# Add text annotations
for i in range(len(method_names)):
    rank_list = rankings[method_names[i]]
    for j in range(max_len):
        if j < len(rank_list):
            doc_idx = rank_list[j]
            rel = relevance[doc_idx]
            text_color = 'white' if rel >= 2 else 'black'
            ax.text(j, i, f'D{doc_idx}\n(rel={rel})', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=text_color)

ax.set_xticks(range(max_len))
ax.set_xticklabels([f'Rank {i+1}' for i in range(max_len)])
ax.set_yticks(range(len(method_names)))
ax.set_yticklabels(method_names)
ax.set_title('Ranking Results Heatmap (Document Relevance at Each Rank Position)')

# Legend
legend_patches = [
    mpatches.Patch(color='#27ae60', label='Rel=3 (Highly Relevant)'),
    mpatches.Patch(color='#2ecc71', label='Rel=2 (Relevant)'),
    mpatches.Patch(color='#f1c40f', label='Rel=1 (Somewhat)'),
    mpatches.Patch(color='#e74c3c', label='Rel=0 (Not Relevant)'),
]
ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)

plt.tight_layout()
plt.savefig('figures/fig7_ranking_heatmap.png')
plt.close()
print("  ✓ Figure 7: Ranking heatmap saved")

print("\n  All 7 figures generated in figures/ directory!")
