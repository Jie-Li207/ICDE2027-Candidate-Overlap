import json
from pathlib import Path
import numpy as np
from confirmatory_data import workload
R = Path(__file__).resolve().parent
O = R / 'chance'
O.mkdir(exist_ok=True)
rows = []
for mf in (R / 'candidate_continuation').glob('*_manifest.json'):
    m = json.loads(mf.read_text())
    ds = m['dataset']
    left = np.array(m['sources'])
    right = np.array(m['nested_target_order'])
    w = workload(ds, left, right, right)
    for f in (R / 'candidate_continuation').glob('*' + ds.replace('/', '_') + '*_results.json'):
        rr = json.loads(f.read_text())
        pred = np.load(f.with_suffix('.npz'))
        for row in rr:
            if row['decoder'] not in ['row', 'dense'] or row['query_set'] == 'cross_filter':
                continue
            n = row['targets']
            U = w[row['query_set']][0]
            K = U.sum(1)
            p = pred[f"{row['score']}_{n}_{row['decoder']}"]
            M = np.array([len(np.unique(p[u])) for u in U])
            null = float(2 * np.sum(K * M) / (n * np.sum(K + M)))
            rows.append({k: row[k] for k in ['dataset', 'seed', 'mode', 'score', 'targets', 'decoder', 'query_set']} | dict(observed_micro_f1=row['micro_f1'], null_expected_micro_f1=null, excess_micro_f1=row['micro_f1'] - null))
(O / 'all_results.json').write_text(json.dumps(rows, indent=2))
out = []
for ds in sorted({r['dataset'] for r in rows}):
    for n in [256, 1024]:
        rr = [r for r in rows if r['dataset'] == ds and r['mode'] == 'initial' and (r['score'] == 'depth2') and (r['targets'] == n) and (r['query_set'] == 'coarse')]
        v = {(r['seed'], r['decoder']): r for r in rr}
        seeds = sorted({r['seed'] for r in rr})
        out.append(dict(dataset=ds, targets=n, **{k: float(np.mean([v[s, 'dense'][k] - v[s, 'row'][k] for s in seeds])) for k in ['observed_micro_f1', 'null_expected_micro_f1', 'excess_micro_f1']}))
(O / 'primary_setting_summary.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
