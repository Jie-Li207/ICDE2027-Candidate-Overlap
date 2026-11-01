import json
from pathlib import Path
import numpy as np
from confirmatory_data import workload
R = Path(__file__).resolve().parent
for f in sorted((R / 'candidate_continuation').glob('*_manifest.json')):
    m = json.loads(f.read_text())
    left = np.array(m['sources'])
    right = np.array(m['nested_target_order'])
    w = workload(m['dataset'], left, right, right)
    assert json.loads(json.dumps({q: x[2] for q, x in w.items()})) == m['query_keys']
    arrays = dict(left=left, right=right)
    for q, (U, V, keys) in w.items():
        arrays[q + '_U'] = U
        arrays[q + '_V'] = V
    np.savez_compressed(f.with_name(f.name.replace('_manifest.json', '_workload.npz')), **arrays)
    print('EXPORTED', m['dataset'])
