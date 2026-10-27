import json, itertools, collections
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent
O = R / 'revision_v36'
O.mkdir(exist_ok=True)

def expected(p, gold, K):
    n = len(p)
    a = np.sum(p == gold)
    b = np.sum(np.isin(p, gold) & (p != gold))
    return float(a / n + b * (K - 1) / (n * (n - 1)))

def fixtures():
    count = 0
    maxerr = 0.0
    for n in [2, 3, 4]:
        g = np.arange(n)
        for p in itertools.permutations(range(n + 1), n):
            p = np.array(p)
            for K in range(1, n + 1):
                values = []
                for U in itertools.combinations(range(n), K):
                    values.append(len(set(p[list(U)]) & set(g[list(U)])) / K)
                err = abs(float(np.mean(values)) - expected(p, g, K))
                assert err < 1e-12
                maxerr = max(err, maxerr)
                count += 1
    return dict(passed=True, map_size_cases=count, max_error=maxerr)

def main():
    rows = []
    for folder in ['revision_v34', 'revision_v35', 'revision_v38', 'revision_v39']:
        for path in sorted((R / folder).glob('*_results.json')):
            data = json.loads(path.read_text())
            first = data['rows'][0]
            base = {k: first[k] for k in ['dataset', 'seed', 'cohort', 'scorer', 'block'] if k in first}
            base['evidence_folder'] = folder
            stem = first['dataset'].replace('/', '_') + '_' + first['cohort'] if folder in ['revision_v34', 'revision_v39'] else first['dataset'] + f"_block{first['block']}"
            z = np.load(R / folder / (stem + '_workload.npz'))
            wm = json.loads((R / folder / (stem + '_workload.json')).read_text())
            pred = np.load(path.with_suffix('.npz'))
            gold = z['gold']
            n = len(gold)
            for regime in ['full', 'oracle_closed']:
                p = pred[regime + '_dense_p']
                assert (p >= 0).all() and len(set(p)) == n
                c = pred[regime + '_dense_counts']
                a = int(np.sum(p == gold))
                b = int(np.sum(np.isin(p, gold) & (p != gold)))
                for row in [r for r in data['rows'] if r['id'] == regime + '_dense' and r['family'] not in ['all', 'original_cross_filter']]:
                    ix = [i for i, q in enumerate(wm['queries']) if q['family'] == row['family'] and str(q['bin']) == row['bin']]
                    K = z['U'][ix].sum(1)
                    assert np.array_equal(c[ix, 0] + c[ix, 1], K)
                    assert np.array_equal(c[ix, 0] + c[ix, 2], K)
                    null = float(a / n + b * np.sum(K * (K - 1)) / (n * (n - 1) * np.sum(K)))
                    rows.append(dict(**base, regime=regime, family=row['family'], bin=row['bin'], query_count=len(ix), identity_correct=a, wrong_inside_true_targets=b, wrong_outside_true_targets=n - a - b, mean_selectivity=float(np.mean(K / n)), observed_f1=row['micro_f1'], cardinality_null_f1=null, semantic_excess=row['micro_f1'] - null))
    (O / 'results.json').write_text(json.dumps(rows, indent=2))
    (O / 'fixtures.json').write_text(json.dumps(fixtures(), indent=2))
    print('MECHANISM', len(rows))
if __name__ == '__main__':
    main()
