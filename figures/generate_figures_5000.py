"""Generate publication-quality figures for the thesis (5000 steps version)."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# IEEE-style configuration
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (3.5, 2.5),
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

# Colorblind-safe palette
COLORS = {
    'blue': '#4C78A8',
    'orange': '#F58518',
    'green': '#54A24B',
    'red': '#E45756',
    'purple': '#B279A2',
    'brown': '#9D755D',
    'gray': '#797979',
}

OUTPUT_DIR = Path(__file__).parent


def fig1_ablation_summary():
    """Figure 1: Component ablation summary (bar chart) - 5000 steps."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    configs = ['Trans', 'SwiGLU', 'Longer', 'Longer\n+Causal', 'RM-5-2', 'RM-1-2', 'RM-5-10',
               'Full', 'FFNOnly', 'None', 'NoRoPE', 'RoPE']
    # 5000 steps data
    aucs = [0.7351, 0.7368, 0.7352, 0.7352, 0.7347, 0.7415, 0.7348,
            0.7358, 0.7357, 0.7352, 0.7352, 0.7352]
    groups = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    group_colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['purple']]
    colors = [group_colors[g] for g in groups]

    bars = ax.bar(range(len(configs)), aucs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('AUC')
    ax.set_ylim(0.73, 0.745)
    ax.axhline(y=0.7415, color=COLORS['red'], linestyle='--', linewidth=1, label='Best (0.7415)')

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{auc:.4f}', ha='center', va='bottom', fontsize=6)

    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
    ax.set_title('(a) Component Ablation (5000 steps)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_ablation_summary_5000.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig1_ablation_summary_5000.png', bbox_inches='tight')
    plt.close()
    print("Saved fig1_ablation_summary_5000")


def fig2_scaling_dim():
    """Figure 2: Embedding dimension scaling law - 5000 steps."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    dims = [32, 64, 128, 256]
    aucs = [0.7348, 0.7352, 0.7351, 0.7352]  # 5000 steps
    params = [2.83, 2.84, 6.45, 16.02]

    ax2 = ax.twinx()

    line1 = ax.plot(dims, aucs, 'o-', color=COLORS['blue'], linewidth=2,
                    markersize=8, label='AUC', zorder=3)
    ax.fill_between(dims, aucs, alpha=0.1, color=COLORS['blue'])

    line2 = ax2.plot(dims, params, 's--', color=COLORS['orange'], linewidth=2,
                     markersize=7, label='Parameters', zorder=2)

    ax.set_xlabel('Embedding Dimension (d_model)')
    ax.set_ylabel('AUC', color=COLORS['blue'])
    ax2.set_ylabel('Parameters (M)', color=COLORS['orange'])
    ax.set_ylim(0.733, 0.737)
    ax2.set_ylim(0, 20)

    ax.tick_params(axis='y', labelcolor=COLORS['blue'])
    ax2.tick_params(axis='y', labelcolor=COLORS['orange'])

    ax.axvline(x=64, color=COLORS['red'], linestyle=':', alpha=0.7, label='Sweet spot d=64')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', frameon=True, fancybox=False,
              edgecolor='black', fontsize=8)

    ax.set_title('(b) Embedding Dimension Scaling (5000s)', fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_scaling_dim_5000.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_scaling_dim_5000.png', bbox_inches='tight')
    plt.close()
    print("Saved fig2_scaling_dim_5000")


def fig3_scaling_depth():
    """Figure 3: Model depth scaling law - 5000 steps."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    depths = [1, 2, 4, 8]
    aucs = [0.7367, 0.7352, 0.7353, 0.7360]  # 5000 steps
    params = [2.76, 2.84, 2.99, 3.29]

    ax2 = ax.twinx()

    line1 = ax.plot(depths, aucs, 'o-', color=COLORS['green'], linewidth=2,
                    markersize=8, label='AUC', zorder=3)
    ax.fill_between(depths, aucs, alpha=0.1, color=COLORS['green'])

    line2 = ax2.plot(depths, params, 's--', color=COLORS['orange'], linewidth=2,
                     markersize=7, label='Parameters', zorder=2)

    ax.set_xlabel('Number of HyFormer Blocks')
    ax.set_ylabel('AUC', color=COLORS['green'])
    ax2.set_ylabel('Parameters (M)', color=COLORS['orange'])
    ax.set_ylim(0.734, 0.738)
    ax2.set_ylim(2.5, 3.5)

    ax.tick_params(axis='y', labelcolor=COLORS['green'])
    ax2.tick_params(axis='y', labelcolor=COLORS['orange'])

    ax.axvline(x=1, color=COLORS['red'], linestyle=':', alpha=0.7, label='Best efficiency b=1')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', frameon=True, fancybox=False,
              edgecolor='black', fontsize=8)

    ax.set_title('(c) Model Depth Scaling (5000s)', fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_scaling_depth_5000.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_scaling_depth_5000.png', bbox_inches='tight')
    plt.close()
    print("Saved fig3_scaling_depth_5000")


def fig4_comparison_2000_vs_5000():
    """Figure 4: 2000 vs 5000 steps comparison."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    configs = ['Trans', 'SwiGLU', 'Longer', 'L+Causal', 'RM-1-2', 'RM-5-2', 'Full', 'None', 'b=1', 'b=2']
    auc_2000 = [0.7234, 0.7199, 0.7234, 0.7354, 0.7366, 0.7354, 0.7354, 0.7369, 0.7365, 0.7369]
    auc_5000 = [0.7351, 0.7368, 0.7352, 0.7352, 0.7415, 0.7347, 0.7358, 0.7352, 0.7367, 0.7352]

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, auc_2000, width, label='2000 steps', color=COLORS['blue'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, auc_5000, width, label='5000 steps', color=COLORS['orange'], edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('AUC')
    ax.set_ylim(0.71, 0.745)
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
    ax.set_title('(d) 2000 vs 5000 Steps Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_comparison_5000.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_comparison_5000.png', bbox_inches='tight')
    plt.close()
    print("Saved fig4_comparison_5000")


def fig5_pareto():
    """Figure 5: Accuracy vs Efficiency Pareto frontier - 5000 steps."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    configs = [
        (2.83, 0.7348, 'd=32'),
        (2.84, 0.7352, 'd=64'),
        (6.45, 0.7351, 'd=128'),
        (16.02, 0.7352, 'd=256'),
        (2.76, 0.7367, 'b=1 (best eff)'),
        (2.84, 0.7352, 'b=2'),
        (2.99, 0.7353, 'b=4'),
        (3.29, 0.7360, 'b=8'),
        (2.84, 0.7406, 'E10 Final'),
        (2.84, 0.7415, 'E2 Best'),
    ]

    for i, (p, a, l) in enumerate(configs):
        if 'Best' in l:
            color = COLORS['red']
            size = 120
        elif 'Final' in l:
            color = COLORS['purple']
            size = 100
        else:
            color = COLORS['blue']
            size = 50
        ax.scatter(p, a, s=size, color=color, edgecolor='black', linewidth=0.5, zorder=3)
        ax.annotate(l, (p, a), textcoords="offset points", xytext=(5, 5),
                   fontsize=6, ha='left')

    ax.set_xlabel('Parameters (Millions)')
    ax.set_ylabel('AUC')
    ax.set_ylim(0.733, 0.743)
    ax.set_xlim(2.5, 17)

    ax.set_title('(e) Accuracy vs Model Size (5000s)', fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_pareto_5000.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig5_pareto_5000.png', bbox_inches='tight')
    plt.close()
    print("Saved fig5_pareto_5000")


def fig6_seed_robustness():
    """Figure 6: Seed robustness distribution."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    seeds = [42, 3507, 114514, 4615, 2026, 455711, 734613, 734087, 544634, 978088]
    aucs = [0.741516, 0.740195, 0.741571, 0.741239, 0.739328,
            0.738855, 0.741497, 0.740287, 0.740952, 0.740618]
    labels = ['42', '3507', '114514', '4615', '2026', 'R1', 'R2', 'R3', 'R4', 'R5']

    colors = [COLORS['blue'] if s < 1000000 else COLORS['orange'] for s in seeds]

    ax.scatter(range(len(seeds)), aucs, s=80, c=colors, edgecolor='black', linewidth=0.5, zorder=3)
    ax.axhline(y=0.740606, color=COLORS['red'], linestyle='--', linewidth=1, label=f'Mean=0.7406')
    ax.axhline(y=0.740606 + 0.000945, color=COLORS['red'], linestyle=':', alpha=0.5)
    ax.axhline(y=0.740606 - 0.000945, color=COLORS['red'], linestyle=':', alpha=0.5)

    ax.fill_between(range(-1, 11), 0.740606 - 0.000945, 0.740606 + 0.000945,
                    alpha=0.1, color=COLORS['red'])

    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('AUC')
    ax.set_ylim(0.738, 0.742)
    ax.set_xlim(-0.5, 9.5)

    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black', fontsize=8)
    ax.set_title('(f) Seed Robustness (CV=0.128%)', fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_seed_robustness.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig6_seed_robustness.png', bbox_inches='tight')
    plt.close()
    print("Saved fig6_seed_robustness")


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig1_ablation_summary()
    fig2_scaling_dim()
    fig3_scaling_depth()
    fig4_comparison_2000_vs_5000()
    fig5_pareto()
    fig6_seed_robustness()

    print(f"\nAll figures saved to {OUTPUT_DIR}")
