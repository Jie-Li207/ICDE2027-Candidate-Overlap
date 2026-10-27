import argparse, json, hashlib, collections
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent
O = R / 'confirmatory_v27'

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--allow-incomplete', action='store_true')
    opt = a.parse_args()
    counts = collections.Counter()
    checked = 0
    cache = {}
    signatures = set()
    for f in sorted((O / 'results').glob('*.json')):
        x = json.loads(f.read_text())
        r = x['rows'][0]
        counts[r['learner']] += 1
        key = (r['dataset'], r['block'], r['regime'])
        if key not in cache:
            cache[key] = np.load(O / (r['dataset'].replace('/', '_') + f"_block{r['block']}_{r['regime']}_workload.npz"))
        z = cache[key]
        pred = np.load(f.with_suffix('.npz'))
        for row in x['rows']:
            sig = tuple((row[k] for k in ['dataset', 'learner', 'seed', 'mode', 'block', 'regime', 'score', 'decoder', 'query_set']))
            assert sig not in signatures
            signatures.add(sig)
            p = pred[row['decoder']]
            q = row['query_set']
            U = z[q + '_U']
            V = z[q + '_V']
            g = z['gold']
            tp = fp = fn = 0
            for u, v in zip(U, V):
                predicted = set((int(j) for j in p[u] if v[j]))
                truth = set((int(j) for j in g[u] if v[j]))
                tp += len(predicted & truth)
                fp += len(predicted - truth)
                fn += len(truth - predicted)
            for k, val in [('TP', tp), ('FP', fp), ('FN', fn)]:
                assert row[k] == val, (f, k, row[k], val)
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
            assert abs(row['micro_f1'] - f1) < 1e-12
            assert row['key_collisions'] == len(p) - len(set(map(int, p)))
            assert abs(row['entity_accuracy'] - float((p == g).mean())) < 1e-12
            checked += 1
    expected = {'DESAlign': 480, 'DESAlign+TFP': 240, 'visual_zero_shot': 160, 'visual_anchor_equal': 160, 'anchor128': 80, 'CDMEA-50': 192, 'CDMEA-200': 192}
    if not opt.allow_incomplete:
        assert dict(counts) == expected, (counts, expected)
    for f in O.glob('*_identities.json'):
        m = json.loads(f.read_text())
        train = set(np.array(m['train']).ravel())
        dev = set(np.array(m['dev']).ravel())
        test = set(np.array(m['test']).ravel())
        assert not train & dev and (not train & test) and (not dev & test)
        assert m['sealed_hash_verified']
        assert len(np.unique(np.array(m['test'])[:, 0])) == len(m['test'])
    rec = dict(independent_metric_rows_checked=checked, condition_counts=dict(counts), expected_condition_counts=expected, complete=dict(counts) == expected, identity_boundaries_verified=True)
    (R / 'recorded_result_validation.json').write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
if __name__ == '__main__':
    main()
