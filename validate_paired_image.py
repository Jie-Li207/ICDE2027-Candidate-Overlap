import json
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent
O = R / 'paired_image'

def main():
    checked = 0
    summary = []
    strata = []
    for mf in sorted(O.glob('*_manifest.json')):
        m = json.loads(mf.read_text())
        strata.append(dict(dataset=m['dataset'], available=m['safe_shared_image_pairs'], selected=len(m['selected']), status=m['status']))
        if m['status'] != 'evaluated':
            continue
        tag = m['dataset'].replace('/', '_')
        z = np.load(O / (tag + '_workload.npz'))
        pred = np.load(O / (tag + '_results.npz'))
        rows = json.loads((O / (tag + '_results.json')).read_text())
        cache = {}
        n = len(m['selected'])
        for r in rows:
            size = r['targets']
            q = r['query_set']
            if (size, q) not in cache:
                order = np.random.default_rng(33002 + size).permutation(size)
                cache[size, q] = (z[q + '_U'], z[q + '_V'][:, order], np.argsort(order)[:n])
            U, V, g = cache[size, q]
            p = pred[f"{r['learner']}_{r['mode']}_{size}_{r['decoder']}"]
            tp = fp = fn = 0
            for u, v in zip(U, V):
                a = set(map(int, p[u][v[p[u]]]))
                t = set(map(int, g[u][v[g[u]]]))
                tp += len(a & t)
                fp += len(a - t)
                fn += len(t - a)
            assert [tp, fp, fn] == [r['TP'], r['FP'], r['FN']]
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
            assert abs(f1 - r['micro_f1']) < 1e-12
            assert abs(float((p == g).mean()) - r['entity_accuracy']) < 1e-12 and r['key_collisions'] == n - len(set(map(int, p)))
            checked += 1
        for learner, mode in sorted({(r['learner'], r['mode']) for r in rows}):
            values = {}
            for size in m['sizes']:
                rr = {r['decoder']: r for r in rows if r['learner'] == learner and r['mode'] == mode and (r['targets'] == size) and (r['query_set'] == 'coarse')}
                values[str(size)] = {k: rr[k]['micro_f1'] for k in ['row', 'dense']}
                values[str(size)]['assignment_advantage'] = rr['dense']['micro_f1'] - rr['row']['micro_f1']
                values[str(size)]['row_identity_accuracy'] = rr['row']['entity_accuracy']
            summary.append(dict(dataset=m['dataset'], n=n, learner=learner, mode=mode, values=values))
    out = dict(passed=True, checked_rows=checked, strata=strata, scope='Secondary boundary analysis; no primary gate changed')
    (O / 'validation.json').write_text(json.dumps(out, indent=2))
    (O / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(out, indent=2))
    for r in summary:
        if r['mode'] == 'initial' and r['learner'] == 'DESAlign_depth2':
            print(json.dumps(r))
if __name__ == '__main__':
    main()
