import json, collections, statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_confirmatory import bootstrap
R = Path(__file__).resolve().parent
P = R / 'paper'
F = P / 'figures'
F.mkdir(exist_ok=True)
T = P / 'tables'
T.mkdir(exist_ok=True)
DS = ['FBDB15K/norm', 'FBYG15K/norm', 'DBP15K/zh_en', 'DBP15K/fr_en']
SHORT = dict(zip(DS, ['FB--DB', 'FB--YG', 'ZH--EN', 'FR--EN']))
colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
rows = []
for f in (R / 'confirmatory/results').glob('*.json'):
    rows.extend(json.loads(f.read_text())['rows'])
cand = json.loads((R / 'candidate_continuation/all_results.json').read_text())
summary = json.loads((R / 'confirmatory/summary.json').read_text())
paired = json.loads((R / 'confirmatory/paired.json').read_text())

def sel(rr=rows, **kw):
    return [r for r in rr if all((r.get(k) == v for k, v in kw.items()))]

def mean(rr, field):
    return float(np.mean([r[field] for r in rr]))

def save(fig, name):
    fig.savefig(F / (name + '.pdf'), bbox_inches='tight', pad_inches=0.035)
    fig.savefig(F / (name + '.svg'), bbox_inches='tight', pad_inches=0.035)
    fig.savefig(F / (name + '.png'), dpi=160, bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)

def table(name, header, body, cols):
    text = '\\begin{tabular}{' + cols + '}\n\\toprule\n' + header + ' \\\\\n\\midrule\n' + '\n'.join((' & '.join(map(str, r)) + ' \\\\' for r in body)) + '\n\\bottomrule\n\\end{tabular}\n'
    (T / (name + '.tex')).write_text(text)

def fmt(x):
    return f'{100 * x:.1f}'
fig, axes = plt.subplots(1, 4, figsize=(7.05, 2.25), sharey=True)
for ax, ds in zip(axes, DS):
    for mode, col, mark in [('initial', colors[0], 'o'), ('full', colors[1], 's'), ('noimage', colors[2], '^')]:
        rr = sel(cand, dataset=ds, mode=mode, score='depth2', query_set='coarse')
        sizes = sorted({r['targets'] for r in rr})
        means = []
        lo = []
        hi = []
        for n in sizes:
            aa = {r['seed']: r['micro_f1'] for r in rr if r['targets'] == n and r['decoder'] == 'dense'}
            bb = {r['seed']: r['micro_f1'] for r in rr if r['targets'] == n and r['decoder'] == 'row'}
            d = np.array([aa[s] - bb[s] for s in aa]) * 100
            means.append(d.mean())
            lo.append(d.min())
            hi.append(d.max())
        ax.plot(np.array(sizes) / 256, means, marker=mark, ms=3, color=col, label=mode)
        ax.fill_between(np.array(sizes) / 256, lo, hi, alpha=0.13, color=col)
    ax.axhline(0, color='0.5', lw=0.6)
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 4, 16, 64], ['1', '4', '16', '64'])
    ax.set_title(SHORT[ds].replace('--', '–'))
    ax.set_xlabel('Targets / sources')
    ax.grid(alpha=0.2)
axes[0].set_ylabel('Assignment − row F1 (pp)')
axes[1].legend(frameon=False, loc='upper right')
fig.tight_layout(w_pad=0.65)
save(fig, 'candidate')
fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.45), sharey=True)
for ax, regime, a, b, title in [(axes[0], 'closed', 'ot500_row', 'ot500_dense', 'Closed: OT-entry assignment − OT row'), (axes[1], 'expanded', 'row', 'dense', 'Expanded: raw assignment − raw row')]:
    for off, metric, col, label in [(-0.12, 'entity_accuracy', colors[0], 'Identity accuracy'), (0.12, 'micro_f1', colors[1], 'Query F1')]:
        for j, ds in enumerate(DS):
            rr = sel(paired, dataset=ds, learner='DESAlign', mode='initial', regime=regime, score='depth2', query_set='coarse', first=a, second=b)
            assert len(rr) == 1
            v = rr[0][metric + '_delta']
            x = 100 * v['mean']
            ax.errorbar(x, j + off, xerr=[[100 * (v['mean'] - v['lower'])], [100 * (v['upper'] - v['mean'])]], fmt='o', ms=4, color=col, label=label if j == 0 else None, capsize=2)
    ax.axvline(0, color='0.4', lw=0.7)
    ax.set_yticks(range(4), [SHORT[d].replace('--', '–') for d in DS])
    ax.set_xlabel('Paired change (percentage points)')
    ax.set_title(title, fontsize=9)
    ax.grid(axis='x', alpha=0.2)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout(w_pad=1)
save(fig, 'paired')
fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.35), sharey=True)
modal = []
for ax, q, title in zip(axes, ['coarse', 'value', 'cross_filter'], ['Relation existence', 'Relation + object', 'Cross-source filter']):
    for j, ds in enumerate(DS):
        for off, regime, col in [(-0.13, 'closed', colors[0]), (0.13, 'expanded', colors[1])]:
            rr = sel(dataset=ds, learner='DESAlign', score='depth2', decoder='dense', query_set=q, regime=regime)
            v = {(r['seed'], r['block'], r['mode']): r['micro_f1'] for r in rr}
            seeds = sorted({r['seed'] for r in rr})
            bs = bootstrap([[v[s, b, 'full'] - v[s, b, 'initial'] for b in range(4)] for s in seeds])
            modal.append(dict(dataset=ds, query=q, regime=regime, **bs))
            ax.errorbar(j + off, bs['mean'] * 100, yerr=[[100 * (bs['mean'] - bs['lower'])], [100 * (bs['upper'] - bs['mean'])]], fmt='o', ms=4, color=col, capsize=2, label=regime if j == 0 else None)
    ax.set_xticks(range(4), [SHORT[d].replace('--', '–') for d in DS], rotation=30)
    ax.axhline(0, color='0.4', lw=0.7)
    ax.grid(axis='y', alpha=0.2)
    ax.set_title(title)
axes[0].set_ylabel('Full − initial query F1 (pp)')
axes[2].legend(frameon=False)
fig.tight_layout(w_pad=0.7)
save(fig, 'visual')
(P / 'modal_deltas.json').write_text(json.dumps(modal, indent=2))
fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.4), sharey=True)
for ax, q, title in zip(axes, ['coarse', 'value', 'cross_filter'], ['Relation existence', 'Relation + object', 'Cross-source filter']):
    for j, (mode, dec, label, col) in enumerate([('initial', 'row', 'Raw row', colors[0]), ('initial', 'dense', 'Raw assignment', colors[1]), ('initial', 'ot500_row', 'OT row', colors[2]), ('initial', 'ot500_dense', 'OT-entry assignment', colors[3])]):
        vals = [mean(sel(dataset=ds, learner='DESAlign', mode=mode, score='depth2', regime='closed', decoder=dec, query_set=q), 'micro_f1') * 100 for ds in DS]
        ax.plot(range(4), vals, marker=['o', 's', '^', 'D'][j], ms=3, label=label, color=col)
    ax.set_xticks(range(4), [SHORT[d].replace('--', '–') for d in DS], rotation=30)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(alpha=0.2)
axes[0].set_ylabel('Query F1 (%)')
axes[1].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.32), frameon=False)
fig.tight_layout(w_pad=0.7)
save(fig, 'granularity')
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.75), gridspec_kw={'width_ratios': [1, 1.2]})
A = np.array([[0.44, 0.43, 0.13], [0.42, 0.31, 0.27], [0.14, 0.26, 0.6]])
assert np.allclose(A.sum(0), 1) and np.allclose(A.sum(1), 1)
ax = axes[0]
ax.imshow(A, cmap='Blues', vmin=0, vmax=0.65)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{A[i, j]:.2f}', ha='center', va='center', fontsize=9, color='white' if A[i, j] > 0.4 else 'black')
ax.set_xticks(range(3), ['t1', 't2', 't3'])
ax.set_yticks(range(3), ['s1', 's2', 's3'])
ax.set_title('Unit row/column sums', fontsize=9)
ax = axes[1]
ax.axis('off')
ax.text(0, 0.9, 'Row maximum: (t1, t1, t3)\nTarget coverage: 2/3\n\nAssignment: (t2, t1, t3)\nTarget coverage: 3/3\n\nUnary query: {s1, s2}\nRow answer: {t1}', fontsize=9, va='top')
fig.tight_layout(w_pad=0.2)
save(fig, 'mechanism')
body = []
for ds in DS:
    m = json.loads((R / 'confirmatory' / (ds.replace('/', '_') + '_identities.json')).read_text())
    body.append([SHORT[ds], f"{m['eligible_count']:,}", f"{len(m['test']):,}", f"{len(m['gallery']):,}", 3 if ds in [DS[0], DS[2]] else 2])
table('data', 'Corpus & Eligible & Sealed pool & Gallery & Seeds', body, 'lrrrr')
body = []
for learner, score, label in [('visual_zero_shot', 'visual_cosine', 'Visual cosine'), ('anchor128', 'anchor_cosine', 'Anchor graph'), ('visual_anchor_equal', 'equal_cosine', 'Equal mixture'), ('DESAlign', 'depth0', 'DES base'), ('DESAlign', 'depth2', 'DES depth 2'), ('DESAlign+TFP', 'tfp_cosine', 'DES + TFP'), ('CDMEA-50', 'CDI_CSLS', 'CDMEA 50'), ('CDMEA-200', 'CDI_CSLS', 'CDMEA 200')]:
    if learner.startswith('visual') or learner == 'anchor128':
        avail = sorted({r['score'] for r in rows if r['learner'] == learner})
        assert len(avail) == 1
        score = avail[0]
    vals = []
    for ds in DS:
        rr = sel(dataset=ds, learner=learner, score=score, mode='initial', regime='closed', decoder='dense', query_set='coarse')
        if not rr and learner == 'anchor128':
            rr = sel(dataset=ds, learner=learner, score=score, regime='closed', decoder='dense', query_set='coarse')
        vals.append('--' if not rr else fmt(mean(rr, 'entity_accuracy')) + '/' + fmt(mean(rr, 'micro_f1')))
    body.append([label] + vals)
table('learners', 'Scorer & FB--DB & FB--YG & ZH--EN & FR--EN', body, 'lrrrr')
body = []
for ds in DS:
    vals = []
    for regime in ['closed', 'expanded']:
        for decoder in ['row', 'dense']:
            rr = sel(dataset=ds, learner='DESAlign', mode='initial', score='depth2', regime=regime, decoder=decoder, query_set='coarse')
            vals.append(fmt(mean(rr, 'entity_accuracy')) + '/' + fmt(mean(rr, 'micro_f1')))
    body.append([SHORT[ds]] + vals)
table('regimes', 'Corpus & Closed row & Closed assign. & Expanded row & Expanded assign.', body, 'lrrrr')
body = []
for ds in DS:
    rr = sel(dataset=ds, learner='DESAlign', mode='initial', score='depth2', regime='closed', decoder='ot500_row', query_set='coarse')
    body.append([SHORT[ds], fmt(1 - mean(rr, 'target_coverage')), f"{max((r['marginal_residual'] for r in rr)):.1e}", sum((r['near_tie_rows'] for r in rr)), len(rr) * 512])
table('soft', 'Corpus & Collision rate (\\%) & Residual & Near ties & Rows', body, 'lrrrr')
direct = []
for f in (R / 'confirmatory/direct').glob('*.json'):
    direct.extend(json.loads(f.read_text()))
body = []
for learner, decision, label in [('target_visual_ridge', 'threshold', 'Visual ridge, threshold'), ('target_visual_ridge', 'source_cardinality_topk', 'Visual ridge, top-$k$'), ('target_graph_propagation', 'threshold', 'Graph propagation, threshold'), ('target_graph_propagation', 'source_cardinality_topk', 'Graph propagation, top-$k$')]:
    body.append([label] + [fmt(mean(sel(direct, dataset=ds, learner=learner, decision=decision, regime='closed', query_set='coarse'), 'micro_f1')) for ds in DS])
table('direct', 'Query predictor & FB--DB & FB--YG & ZH--EN & FR--EN', body, 'lrrrr')
secondary = []
for f in (R / 'candidate_controls').glob('*.json'):
    secondary.extend(json.loads(f.read_text()))
if secondary:
    body = []
    for learner, label in [('visual_zero_shot', 'Visual cosine'), ('anchor128', 'Anchor graph'), ('visual_anchor_equal', 'Equal mixture')]:
        vals = []
        for ds in DS:
            rr = sel(secondary, dataset=ds, learner=learner, mode='initial', query_set='coarse')
            v = {(r['seed'], r['targets'], r['decoder']): r['micro_f1'] for r in rr}
            seeds = sorted({r['seed'] for r in rr})
            g = [np.mean([v[s, n, 'dense'] - v[s, n, 'row'] for s in seeds]) * 100 for n in [256, 1024]]
            vals.append(f'{g[0]:.2f}/{g[1]:.2f}')
        body.append([label] + vals)
    table('candidate_simple', 'Scorer & FB--DB & FB--YG & ZH--EN & FR--EN', body, 'lrrrr')
if (R / 'chance/primary_setting_summary.json').exists():
    chance = json.loads((R / 'chance/primary_setting_summary.json').read_text())
    body = []
    for ds in DS:
        vals = []
        for n in [256, 1024]:
            r = sel(chance, dataset=ds, targets=n)[0]
            vals.extend([f'{r[k] * 100:.2f}' for k in ['observed_micro_f1', 'null_expected_micro_f1', 'excess_micro_f1']])
        body.append([SHORT[ds]] + vals)
    table('chance', 'Corpus & $\\Delta_{256}$ & Null & Excess & $\\Delta_{1024}$ & Null & Excess', body, 'lrrrrrr')
for ds in DS:
    rr = sel(direct, dataset=ds, regime='closed', query_set='coarse')
    print('DIRECT FEASIBILITY', ds, sum((r['joint_query_feasible'] is True for r in rr)), len(rr))
systems = []
for f in (R / 'systems').glob('*.json') if (R / 'systems').exists() else []:
    x = json.loads(f.read_text())
    if x.get('complete'):
        systems.append(x)
if systems:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55))
    for name, col, marker in [('score', colors[0], 'o'), ('dense', colors[1], 's'), ('sparse', colors[2], '^'), ('ot15', colors[3], 'D')]:
        rr = sorted([r for r in systems if r['dataset'] == DS[0] and r['regime'] == 'closed'], key=lambda r: r['n'])
        x = [r['n'] for r in rr]
        times = [r[name] if name in ['score', 'ot15'] else r['decoders'][name] for r in rr]
        y = [v['median_s'] for v in times]
        axes[0].plot(x, y, label=name, color=col, marker=marker, ms=4)
        axes[0].fill_between(x, [v['q25_s'] for v in times], [v['q75_s'] for v in times], color=col, alpha=0.15)
    axes[0].set_xscale('log', base=2)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Source / target entities')
    axes[0].set_ylabel('Median time (seconds)')
    axes[0].set_title('FB–DB, 2 CPU threads')
    axes[0].legend(ncol=2, frameon=False)
    for ds, col in zip(DS, colors):
        rr = sorted([r for r in systems if r['dataset'] == ds and r['regime'] == 'closed'], key=lambda r: r['n'])
        axes[1].plot([r['n'] for r in rr], [r['peak_process']['peak_working_set_bytes'] / 1024 ** 3 for r in rr], label=SHORT[ds].replace('--', '–'), marker='o', ms=3, color=col)
    axes[1].set_xscale('log', base=2)
    axes[1].set_xlabel('Source / target entities')
    axes[1].set_ylabel('Process peak working set (GiB)')
    axes[1].legend(frameon=False, ncol=2)
    axes[1].set_title('Includes loaded representations')
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, 'systems')
    body = []
    for r in sorted(systems, key=lambda r: (DS.index(r['dataset']), r['regime'], r['n'])):
        if r['regime'] == 'expanded' or r['n'] == max((z['n'] for z in systems if z['dataset'] == r['dataset'] and z['regime'] == 'closed')):
            body.append([SHORT[r['dataset']], f"{r['n']}x{r['m']}", f"{r['score']['median_s']:.3f}", f"{r['decoders']['dense']['median_s']:.3f}", f"{r['decoders']['stable']['median_s']:.3f}", f"{r['decoders']['sparse']['median_s']:.3f}", f"{r['sql_queries']['median_s'] * 1000:.2f}"])
    table('systems', 'Corpus & Shape & Score (s) & Dense (s) & Stable (s) & Sparse (s) & SQL (ms)', body, 'llrrrrr')
if (R / 'paired_image/summary.json').exists():
    boundary = json.loads((R / 'paired_image/summary.json').read_text())
    body = []
    for ds in DS:
        rr = sel(boundary, dataset=ds, learner='DESAlign_depth2')
        if not rr:
            continue
        n = rr[0]['n']
        vals = []
        for mode in ['initial', 'full', 'noimage']:
            r = sel(rr, mode=mode)[0]
            vals.append('/'.join((f"{r['values'][str(k)]['assignment_advantage'] * 100:.2f}" for k in [n, 4 * n])))
        body.append([SHORT[ds], n] + vals)
    table('shared_images', 'Corpus & Sources & Initial & Restored & No image', body, 'lrrrr')
evidence = dict(score_conditions=sum((1 for _ in (R / 'confirmatory/results').glob('*.json'))), metric_rows=len(rows), learner_conditions=dict(collections.Counter((r['learner'] for r in rows))), candidate_rows=len(cand), candidate_gate=json.loads((R / 'candidate_continuation/primary_gate.json').read_text()), fidelity_cases=12, fidelity_changed_rows=sum((json.loads(f.read_text())['changed_row_decisions'] for f in (R / 'tfp_fidelity').glob('*.json'))), systems_complete=len(systems))
(P / 'evidence_snapshot.json').write_text(json.dumps(evidence, indent=2))
print(json.dumps(evidence, indent=2))
