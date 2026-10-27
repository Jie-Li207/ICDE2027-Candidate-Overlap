import json, collections
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent

def main():
    checked = collections.Counter()
    cache = {}
    signatures = set()
    for folder, pattern in [('candidate_v29', '*_results.json'), ('candidate_v31', '*.json')]:
        for f in sorted((R / folder).glob(pattern)):
            if f.name == 'all_results.json':
                continue
            rows = json.loads(f.read_text())
            pred = np.load(f.with_suffix('.npz'))
            for r in rows:
                ds = r['dataset']
                n = r['targets']
                q = r['query_set']
                signature = (folder, ds, r['seed'], r.get('score'), r.get('learner'), r['mode'], n, r['decoder'], q)
                assert signature not in signatures
                signatures.add(signature)
                if (ds, n, q) not in cache:
                    z = np.load(R / 'candidate_v29' / (ds.replace('/', '_') + '_workload.npz'))
                    order = np.random.default_rng(29002 + n).permutation(n)
                    g = np.argsort(order)[:256]
                    U = z[q + '_U']
                    V = z[q + '_V'][:, order]
                    truth = [set(map(int, g[u][v[g[u]]])) for u, v in zip(U, V)]
                    cache[ds, n, q] = (U, V, g, truth)
                U, V, g, truth = cache[ds, n, q]
                key = f"{r['score']}_{n}_{r['decoder']}" if folder == 'candidate_v29' else f"{r['learner']}_{r['mode']}_{n}_{r['decoder']}"
                p = pred[key]
                tp = fp = fn = 0
                for u, v, t in zip(U, V, truth):
                    a = set(map(int, p[u][v[p[u]]]))
                    tp += len(a & t)
                    fp += len(a - t)
                    fn += len(t - a)
                assert [tp, fp, fn] == [r['TP'], r['FP'], r['FN']], (f, signature)
                f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
                assert abs(f1 - r['micro_f1']) < 1e-12
                assert r['key_collisions'] == len(p) - len(set(map(int, p))) and abs(r['entity_accuracy'] - float((p == g).mean())) < 1e-12
                checked[folder] += 1
    assert dict(checked) == {'candidate_v29': 5040, 'candidate_v31': 2100}, checked
    out = dict(passed=True, independent_rows=dict(checked), checks=['TP/FP/FN', 'microF1', 'identity_accuracy', 'key_collisions', 'unique_conditions'], scope='Recomputation of recorded outcomes; not a new held-out experiment')
    (R / 'candidate_recorded_validation.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
