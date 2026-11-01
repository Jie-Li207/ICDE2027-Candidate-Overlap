import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import findfont
R = Path(__file__).resolve().parent
O = R / 'query_scope'
F = R / 'paper' / 'figures'
F.mkdir(parents=True, exist_ok=True)
A = json.loads((O / 'aggregate.json').read_text())
try:
    findfont('Times New Roman', fallback_to_default=False)
    font = 'Times New Roman'
except ValueError:
    font = 'STIXGeneral'
plt.rcParams.update({'font.family': font, 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.spines.top': False, 'axes.spines.right': False})
KS = [1, 4, 16, 64]
COLORS = ['#296E9C', '#B85E31', '#227549']
DATASETS = ['FBDB15K/norm', 'FBYG15K/norm', 'DBP15K/zh_en']
SHORT = ['FB--DB', 'FB--YG', 'ZH--EN']

def get(ds, scorer, ret, k, metric):
    a = next((x for x in A if x['dataset'] == ds and x['scorer'] == scorer and (x['retriever'] == ret) and (x['k'] == k)))
    return a['metrics'][metric]

def draw(ax, ds, scorer, ret, metric, color, style='-', marker='o', label=None, band=True):
    q = [get(ds, scorer, ret, k, metric) for k in KS]
    ax.plot(KS, [100 * v['mean'] for v in q], color=color, linestyle=style, marker=marker, markersize=4, lw=1.4, label=label)
    if band:
        ax.fill_between(KS, [100 * v['lo'] for v in q], [100 * v['hi'] for v in q], color=color, alpha=0.1, lw=0)
fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.8))
ax = axes[0, 0]
for ds, c, label in zip(DATASETS, COLORS, SHORT):
    draw(ax, ds, 'DESAlign_depth2', 'csls10', 'pool_local_changed_fraction', c, label=label)
    draw(ax, ds, 'DESAlign_depth2', 'csls10', 'edge_local_changed_fraction', c, '--', 's', band=False)
ax.set_title('(a) Graph answers depend on scope', loc='left', fontweight='bold')
ax.set_ylabel('Changed answers (%)')
ax.set_ylim(-1, 85)
ax.legend(frameon=False, loc='upper right', ncol=1)
ax.text(0.03, 0.96, 'Solid: pool  /  dashed: edge', transform=ax.transAxes, va='top', fontsize=8, color='black')
ax = axes[0, 1]
for ds, c, label in zip(DATASETS, COLORS, SHORT):
    draw(ax, ds, 'DESAlign_depth2', 'csls10', 'mean_closure_fraction', c, label=label)
ax.set_title('(b) Context for exact edge answers', loc='left', fontweight='bold')
ax.set_ylabel('Sources retained (%)')
ax.set_ylim(-1, 103)
ax.legend(frameon=False, loc='upper left', ncol=1)
ax = axes[1, 0]
for scorer, style, marker in [('ridge128', '-', 'o'), ('visual_zero_label', '--', 's')]:
    for decoder, color in [('edge', COLORS[0]), ('pool', COLORS[1])]:
        draw(ax, 'RetailVisualRAG', scorer, 'cosine', decoder + '_query_f1', color, style, marker, band=scorer == 'ridge128')
ax.set_title('(c) Retail: query quality', loc='left', fontweight='bold')
ax.set_ylabel('Query F1 (%)')
ax.set_ylim(65, 94)
ax.legend(handles=[Line2D([0], [0], color=COLORS[0], label='Edge'), Line2D([0], [0], color=COLORS[1], label='Pool'), Line2D([0], [0], color='black', label='128 labels'), Line2D([0], [0], color='black', ls='--', label='Zero labels')], frameon=False, ncol=2, loc='lower right')
ax = axes[1, 1]
for plan, color, style, label in [('edge_local', COLORS[0], '-', 'Edge, local scope'), ('pool_local', COLORS[1], '-', 'Pool, local scope'), ('pool_fixed', COLORS[2], '--', 'Pool, fixed targets')]:
    draw(ax, 'RetailVisualRAG', 'ridge128', 'cosine', plan + '_changed_fraction', color, style, 'o', label)
ax.set_title('(d) Retail: scope dependence', loc='left', fontweight='bold')
ax.set_ylabel('Changed answers (%)')
ax.set_ylim(-1, 102)
ax.legend(frameon=False, ncol=1, loc='upper right')
for ax in axes.ravel():
    ax.set_xscale('log', base=4)
    ax.set_xticks(KS, labels=[str(k) for k in KS])
    ax.set_xlabel('Candidates per source, k')
    ax.grid(axis='y', color='#DDDDDD', linewidth=0.5)
fig.subplots_adjust(left=0.075, right=0.99, bottom=0.12, top=0.93, wspace=0.25, hspace=0.65)
for extension in ['pdf', 'svg', 'png']:
    fig.savefig(F / ('scope.' + extension), dpi=240, bbox_inches='tight', pad_inches=0.03)
plt.close(fig)
print('SCOPE FIGURE COMPLETE', flush=True)
