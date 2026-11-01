import json, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = Path(__file__).resolve().parent
P = R / 'paper'
O = R / 'revision_summary'
O.mkdir(exist_ok=True)
DS = ['FBDB15K/norm', 'FBYG15K/norm', 'DBP15K/zh_en', 'DBP15K/fr_en']
names = ['FB--DB', 'FB--YG', 'ZH--EN', 'FR--EN']
KS = [1, 2, 4, 8, 16, 32, 64, 128]
rows = []
products = []
for folder, dest in [('retrieved_candidates', rows), ('product_controls', products)]:
    for p in sorted((R / folder).glob('*_results.json')):
        dest.extend(json.loads(p.read_text())['rows'])
twohop = [r for p in sorted((R / 'twohop_queries').glob('*_results.json')) for r in json.loads(p.read_text())['rows']]

def sel(rr, **kw):
    return [r for r in rr if all((r.get(k) == v for k, v in kw.items()))]

def avg(rr, k='micro_f1'):
    return float(np.mean([r[k] for r in rr]))

def table(name, head, body, col):
    (P / 'tables' / name).write_text('\\begin{tabular}{' + col + '}\n\\toprule\n' + head + ' \\\\\n\\midrule\n' + '\n'.join((' & '.join(map(str, r)) + ' \\\\' for r in body)) + '\n\\bottomrule\n\\end{tabular}\n')

def fmt(v):
    return f'{100 * v:.1f}'

def save(fig, n):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(P / 'figures' / (n + '.' + ext), bbox_inches='tight', pad_inches=0.04, dpi=160)
    plt.close(fig)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
summary = {'kg': [], 'products': [], 'transitions': [], 'workloads': []}
body = []
for ds, name in zip(DS, names):
    for retr, label in [('multimodal_depth0', 'MM'), ('visual_zero_label', 'Visual')]:
        a = sel(rows, dataset=ds, cohort='diagnostic256', family='original_coarse', retriever=retr, k=16)
        c = sel(a, decoder='union')
        d = {dec: avg(sel(a, decoder=dec)) for dec in ['row', 'edge', 'union']}
        r = dict(dataset=ds, retriever=retr, edge_recall=avg(c, 'edge_recall'), union_recall=avg(c, 'union_target_recall'), unsupported=avg(c, 'unsupported_edge_fraction'), **d)
        summary['kg'].append(r)
        body.append([name, label, fmt(r['edge_recall']), fmt(r['union_recall']), fmt(d['row']), fmt(d['edge']), fmt(d['union']), fmt(r['unsupported'])])
table('retrieved.tex', 'Corpus & Retriever & Edge rec. & Pool rec. & Row & Edge & Pool & New edges', body, 'llrrrrrr')
fig, axes = plt.subplots(2, 4, figsize=(7.05, 3.9), sharex=True)
for j, (ds, name) in enumerate(zip(DS, names)):
    for retr, col, lab in [('multimodal_depth0', '#0072B2', 'MM'), ('visual_zero_label', '#D55E00', 'Visual')]:
        a = sel(rows, dataset=ds, cohort='diagnostic256', family='original_coarse', retriever=retr)
        axes[0, j].plot(KS, [100 * avg(sel(a, k=k, decoder='edge'), 'edge_recall') for k in KS], color=col, marker='o', ms=2.5, label=lab)
        for dec, style in [('edge', '-'), ('union', '--')]:
            vs = np.array([[100 * r['micro_f1'] for r in sel(a, k=k, decoder=dec)] for k in KS])
            axes[1, j].plot(KS, vs.mean(1), color=col, ls=style, label=lab + ' ' + ('edges' if dec == 'edge' else 'pool'))
            axes[1, j].fill_between(KS, vs.min(1), vs.max(1), color=col, alpha=0.1)
    axes[0, j].set_title(name.replace('--', '-'))
    axes[0, j].set_ylim(0, 100)
    for ax in axes[:, j]:
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 4, 16, 64], ['1', '4', '16', '64'])
        ax.grid(alpha=0.2)
    axes[1, j].set_xlabel('Candidates / source')
axes[0, 0].set_ylabel('True-edge recall (%)')
axes[1, 0].set_ylabel('Query F1 (%)')
axes[0, 0].legend(frameon=False)
axes[1, 1].legend(ncol=4, loc='upper center', bbox_to_anchor=(1.1, -0.37), frameon=False)
fig.subplots_adjust(left=0.08, right=0.995, top=0.93, bottom=0.22, wspace=0.32, hspace=0.24)
save(fig, 'retrieved')
for cohort in ['diagnostic256', 'fresh64']:
    for retr in ['multimodal_depth0', 'visual_zero_label']:
        for dec in ['row', 'edge', 'union']:
            nstrict = decreases = 0
            alltrans = nondecrease = 0
            for ds in DS:
                a = sel(rows, dataset=ds, cohort=cohort, family='original_coarse', retriever=retr, decoder=dec)
                for seed in sorted({r['seed'] for r in a}):
                    s = sorted(sel(a, seed=seed), key=lambda r: r['k'])
                    for x, y in zip(s, s[1:]):
                        alltrans += 1
                        if y['edge_recall'] > x['edge_recall'] + 1e-12:
                            nstrict += 1
                            decreases += y['micro_f1'] < x['micro_f1'] - 1e-12
                        nondecrease += y['edge_recall'] >= x['edge_recall'] - 1e-12
            summary['transitions'].append(dict(cohort=cohort, retriever=retr, decoder=dec, strict_recall_increases=nstrict, f1_decreases=int(decreases), all_transitions=alltrans, nondecreasing_recall=int(nondecrease)))
body = []
for ds in ['Abt-Buy', 'Amazon-GoogleProducts']:
    for scorer, label in [('tfidf_zero_label', 'TF-IDF'), ('ridge128', 'Ridge-128')]:
        a = sel(products, dataset=ds, scorer=scorer, family='all')
        d = {cid: avg(sel(a, id=cid)) for cid in ['oracle_closed_dense', 'full_row', 'full_dense', 'tfidf_k16_row', 'tfidf_k16_edge', 'tfidf_k16_union']}
        summary['products'].append(dict(dataset=ds, scorer=scorer, **d))
        body.append([ds.replace('Amazon-GoogleProducts', 'Amazon-Google'), label] + [fmt(x) for x in d.values()])
table('products.tex', 'Catalogs & Scorer & Oracle & Full row & Full assign & $k$16 row & $k$16 edges & $k$16 pool', body, 'llrrrrrr')
fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25), sharey=True)
for ax, family, title in zip(axes, ['relation', 'relation_object', 'two_hop'], ['Relation', 'Relation + object', 'Two-hop path']):
    for ds, name, col in zip(DS, names, ['#0072B2', '#D55E00', '#009E73', '#CC79A7']):
        source = twohop if family == 'two_hop' else rows
        a = sel(source, dataset=ds, cohort='diagnostic256', family=family, id='oracle_closed_dense')
        b = sel(source, dataset=ds, cohort='diagnostic256', family=family, id='full_dense')
        bins = sorted({int(r['bin']) for r in a})
        assert bins
        values = [100 * (avg(sel(a, bin=str(i))) - avg(sel(b, bin=str(i)))) for i in bins]
        ax.plot(bins, values, marker='o', ms=3, color=col, label=name.replace('--', '-'))
    ax.set_xticks(range(7), ['1', '2.5', '5', '10', '25', '50', '100'])
    ax.set_xlabel('Selectivity bin (%)')
    ax.set_title(title)
    ax.grid(alpha=0.2)
axes[0].set_ylabel('Oracle - full F1 (pp)')
axes[1].legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.43), frameon=False)
fig.subplots_adjust(left=0.08, right=0.995, top=0.86, bottom=0.3, wspace=0.28)
save(fig, 'selectivity')
for folder in ['retrieved_candidates', 'product_controls']:
    for p in sorted((R / folder).glob('*_workload.json')):
        w = json.loads(p.read_text())
        counts = collections.Counter(((q['family'], str(q['bin'])) for q in w['queries']))
        summary['workloads'].append(dict(file=p.name, count=w['query_count'], strata=[dict(family=f, bin=b, n=n) for (f, b), n in counts.items()]))
summary['fresh64_k16'] = [dict(dataset=ds, retriever=retr, **{dec: avg(sel(rows, dataset=ds, cohort='fresh64', family='original_coarse', retriever=retr, k=16, decoder=dec)) for dec in ['row', 'edge', 'union']}) for ds in DS for retr in ['multimodal_depth0', 'visual_zero_label']]
(O / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != 'workloads'}, indent=2))
