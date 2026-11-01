import json, collections
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent
O = R / 'query_diagnostics'
O.mkdir(exist_ok=True)
DS = ['FBDB15K/norm', 'FBYG15K/norm', 'DBP15K/zh_en', 'DBP15K/fr_en']

def dump(n, v):
    (O / n).write_text(json.dumps(v, indent=2))
mechanism = []
for path in sorted((R / 'candidate_continuation').glob('*_results.json')):
    if path.name == 'all_results.json':
        continue
    rows = json.loads(path.read_text())
    z = np.load(path.with_suffix('.npz'))
    for row in rows:
        if row['decoder'] != 'row' or row['query_set'] != 'coarse':
            continue
        n = row['targets']
        key = row['score'] + '_' + str(n)
        a = z[key + '_row']
        b = z[key + '_dense']
        gold = np.argsort(np.random.default_rng(29002 + n).permutation(n))[:len(a)]
        changed = a != b
        corrected = (a != gold) & (b == gold)
        damaged = (a == gold) & (b != gold)
        assert np.sum(corrected) - np.sum(damaged) == np.sum(b == gold) - np.sum(a == gold)
        dense = next((r for r in rows if r['targets'] == n and r['score'] == row['score'] and (r['decoder'] == 'dense') and (r['query_set'] == 'coarse')))
        _, inverse, cnt = np.unique(a, return_inverse=True, return_counts=True)
        mechanism.append(dict(dataset=row['dataset'], seed=row['seed'], mode=row['mode'], score=row['score'], targets=n, sources=len(a), collision_rate=1 - len(set(a)) / len(a), collision_involvement=float(np.mean(cnt[inverse] > 1)), reassigned=float(changed.mean()), corrected=float(corrected.mean()), damaged=float(damaged.mean()), wrong_to_wrong=float(np.mean(changed & ~corrected & ~damaged)), correct_fraction_of_reassignments=float(corrected.sum() / changed.sum()) if changed.sum() else None, query_f1_row=row['micro_f1'], query_f1_dense=dense['micro_f1'], query_delta=dense['micro_f1'] - row['micro_f1']))
dump('mechanism.json', mechanism)
query_rows = []
for folder in ['retrieved_candidates', 'twohop_queries']:
    for ds in DS:
        for cohort in ['diagnostic256', 'fresh64']:
            stem = ds.replace('/', '_') + '_' + cohort
            w = np.load(R / folder / (stem + '_workload.npz'))
            meta = json.loads((R / folder / (stem + '_workload.json')).read_text())
            files = sorted((R / folder).glob(ds.replace('/', '_') + '_seed*_' + cohort + '_results.npz'))
            groups = collections.defaultdict(list)
            for i, q in enumerate(meta['queries']):
                groups[q['family'], str(q['bin'])].append(i)
            for family in {q['family'] for q in meta['queries']}:
                groups[family, 'pooled'] = [i for i, q in enumerate(meta['queries']) if q['family'] == family]
            for (family, bi), ix in sorted(groups.items()):
                truth = [len(set(map(int, w['gold'][w['U'][i]])) & set(np.flatnonzero(w['V'][i]))) for i in ix]
                for cid in ['oracle_closed_dense', 'full_dense', 'multimodal_depth0_k16_edge', 'multimodal_depth0_k16_union', 'visual_zero_label_k16_edge', 'visual_zero_label_k16_union']:
                    c = np.array([np.load(p)[cid + '_counts'][ix] for p in files])
                    tot = c.sum(1)
                    scores = np.divide(2 * tot[:, 0], 2 * tot[:, 0] + tot[:, 1] + tot[:, 2], out=np.ones(len(tot)), where=2 * tot[:, 0] + tot[:, 1] + tot[:, 2] > 0)
                    rng = np.random.default_rng(40001)
                    ss = rng.integers(len(c), size=(2000, len(c)))
                    qq = rng.integers(len(ix), size=(2000, len(ix)))
                    sample = c[ss[:, :, None], qq[:, None, :], :].sum(2)
                    den = 2 * sample[:, :, 0] + sample[:, :, 1] + sample[:, :, 2]
                    vals = np.divide(2 * sample[:, :, 0], den, out=np.ones_like(den, dtype=float), where=den > 0).mean(1)
                    query_rows.append(dict(dataset=ds, cohort=cohort, family=family, bin=bi, decoder=cid, queries=len(ix), mean_gold=float(np.mean(truth)), min_gold=min(truth), max_gold=max(truth), empty_fraction=float(np.mean(np.array(truth) == 0)), f1=float(scores.mean()), lower=float(np.quantile(vals, 0.025)), upper=float(np.quantile(vals, 0.975)), scope='Conditional crossed seed/query bootstrap; predicates overlap and are not a production-query sample'))
dump('query_strata.json', query_rows)
summary = []
for ds in DS:
    a = [r for r in mechanism if r['dataset'] == ds and r['mode'] == 'initial' and (r['score'] == 'depth2')]
    full = max((r['targets'] for r in a))
    for n in [256, 1024, full]:
        rs = [r for r in a if r['targets'] == n]
        summary.append(dict(dataset=ds, targets=n, **{k: float(np.mean([r[k] for r in rs])) for k in ['collision_rate', 'reassigned', 'corrected', 'damaged', 'wrong_to_wrong', 'query_delta']}))
dump('mechanism_summary.json', summary)
print(json.dumps(summary, indent=2))
print('STRATA', len(query_rows))
