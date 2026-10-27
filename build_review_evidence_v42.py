import json, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = Path(__file__).resolve().parent
P = R / 'paper'
(P / 'tables').mkdir(parents=True, exist_ok=True)
(P / 'figures').mkdir(exist_ok=True)
DS = ['FBDB15K/norm', 'FBYG15K/norm', 'DBP15K/zh_en', 'DBP15K/fr_en']
SHORT = dict(zip(DS, ['FB--DB', 'FB--YG', 'ZH--EN', 'FR--EN']))
m = json.loads((R / 'review_v40/mechanism_summary.json').read_text())
q = json.loads((R / 'review_v40/query_strata.json').read_text())

def table(name, header, body, fmt):
    (P / 'tables' / (name + '.tex')).write_text('\\begin{tabular}{' + fmt + '}\n\\toprule\n' + header + ' \\\\\n\\midrule\n' + '\n'.join((' & '.join(x) + ' \\\\' for x in body)) + '\n\\bottomrule\n\\end{tabular}\n')
body = []
for x in m:
    body.append([SHORT[x['dataset']], str(x['targets'])] + [f'{100 * x[k]:.2f}' for k in ['collision_rate', 'reassigned', 'corrected', 'damaged', 'wrong_to_wrong', 'query_delta']])
table('rewrites', 'Corpus & Targets & Collisions & Rewritten & Corrected & Damaged & Wrong$\\to$wrong & $\\Delta F_1$', body, 'lrrrrrrr')
body = []
for ds in DS:
    for fam, label in [('relation', 'Relation'), ('relation_object', 'Rel.--object'), ('two_hop', 'Two-hop')]:
        rows = [x for x in q if x['dataset'] == ds and x['cohort'] == 'diagnostic256' and (x['family'] == fam) and (x['bin'] == 'pooled')]
        x = next((x for x in rows if x['decoder'] == 'full_dense'))
        y = next((x for x in rows if x['decoder'] == 'oracle_closed_dense'))
        body.append([SHORT[ds], label, str(x['queries']), f"{x['mean_gold']:.1f}", f"{x['min_gold']}--{x['max_gold']}", f"{100 * x['empty_fraction']:.0f}", f"{100 * x['f1']:.1f} [{100 * x['lower']:.1f},{100 * x['upper']:.1f}]", f"{100 * y['f1']:.1f} [{100 * y['lower']:.1f},{100 * y['upper']:.1f}]"])
table('query_support', 'Corpus & Family & Queries & Mean $|A^*|$ & Range & Empty (\\%) & Full $F_1$ [95\\% CI] & Oracle $F_1$ [95\\% CI]', body, 'llrrrrrr')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25), sharey=True)
for ax, (fam, label) in zip(axes, [('relation', 'Relation existence'), ('relation_object', 'Relation-object'), ('two_hop', 'Two-hop path')]):
    for ds, col, mark in zip(DS, ['#0072B2', '#D55E00', '#009E73', '#CC79A7'], ['o', 's', '^', 'D']):
        rows = sorted([x for x in q if x['dataset'] == ds and x['cohort'] == 'diagnostic256' and (x['family'] == fam) and (x['bin'] != 'pooled') and (x['decoder'] == 'full_dense')], key=lambda x: int(x['bin']))
        ix = [int(x['bin']) for x in rows]
        ys = [100 * x['f1'] for x in rows]
        ax.plot(ix, ys, label=SHORT[ds].replace('--', '-'), c=col, marker=mark, ms=3)
        ax.fill_between(ix, [100 * x['lower'] for x in rows], [100 * x['upper'] for x in rows], color=col, alpha=0.09)
    ax.set_xticks(range(7), ['1', '2.5', '5', '10', '25', '50', '100'])
    ax.set_ylim(0, 100)
    ax.set_xlabel('Selectivity-bin upper bound (%)')
    ax.set_title(label)
    ax.grid(alpha=0.18)
axes[0].set_ylabel('Full-gallery query F1 (%)')
axes[1].legend(ncol=2, frameon=False, fontsize=7)
fig.tight_layout(pad=0.4, w_pad=0.6)
fig.savefig(P / 'figures/absolute_selectivity.pdf', bbox_inches='tight', pad_inches=0.03)
fig.savefig(P / 'figures/absolute_selectivity.png', dpi=180, bbox_inches='tight', pad_inches=0.03)
plt.close(fig)
allrows = [r for p in (R / 'revision_v41').glob('*_results.json') for r in json.loads(p.read_text())['rows'] if r['family'] == 'original_coarse']
out = []
for ds in DS:
    for cohort in ['diagnostic256', 'fresh64']:
        for decoder in ['row', 'edge', 'union']:
            rs = [x for x in allrows if x['dataset'] == ds and x['cohort'] == cohort and (x['decoder'] == decoder) and (x['k'] == 16)]
            out.append(dict(dataset=ds, cohort=cohort, decoder=decoder, **{k: float(np.mean([x[k] for x in rs])) for k in ['micro_f1', 'edge_recall', 'unsupported_edge_fraction']}))
(R / 'review_v40/fusion_summary.json').write_text(json.dumps(out, indent=2))
print('BUILT two tables, absolute-selectivity figure, fusion summary')
