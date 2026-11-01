import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, time, gc
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
import desalign_baseline as b
import desalign_crosskg as cross
from tfp_numpy import norm
from export_frozen import jobs
from confirmatory_data import workload
from confirmatory_eval import assignment, metrics
from research_queue import available
threadpool_limits(2)
R = Path(__file__).resolve().parent
O = R / 'candidate_controls'
O.mkdir(exist_ok=True)
for ds, seed, _ in jobs():
    dest = O / (ds.replace('/', '_') + f'_seed{seed}.json')
    if dest.exists():
        continue
    while available() < 10 * 1024 ** 3:
        time.sleep(15)
    m = json.loads((R / 'candidate_continuation' / (ds.replace('/', '_') + '_manifest.json')).read_text())
    left = np.array(m['sources'])
    right = np.array(m['nested_target_order'])
    w = workload(ds, left, right, right)
    kgs, full, present, *_ = b.prepare(seed) if ds == 'FBDB15K/norm' else cross.prepare(ds, seed)
    initial = kgs['images_list']
    train = kgs['train_ill']
    n = kgs['ent_num']
    T = np.concatenate([np.loadtxt(b.BASE / ds / f'triples_{s}', dtype=np.int64) for s in [1, 2]])
    edge = np.r_[T[:, [0, 2]], T[:, [2, 0]]]
    A = csr_matrix((np.ones(len(edge), np.float32), (edge[:, 0], edge[:, 1])), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1
    A = A.multiply(1 / np.maximum(np.asarray(A.sum(1)), 1)).tocsr()
    Z0 = np.zeros((n, 128), np.float32)
    for j, (a, c) in enumerate(train):
        Z0[a, j] = Z0[c, j] = 1
    Z = Z0.copy()
    for _ in range(8):
        Z = 0.15 * Z0 + 0.85 * (A @ Z)
    Z = norm(Z)
    anc = Z[left] @ Z[right].T
    rows = []
    pred = {}
    restored = initial.copy()
    restored[present] = full[present]
    for mode, F in [('initial', initial), ('full', restored)]:
        F = norm(F)
        vis = F[left] @ F[right].T
        for learner, S in [('visual_zero_shot', vis), ('anchor128', anc), ('visual_anchor_equal', 0.5 * (vis + anc))]:
            if learner == 'anchor128' and mode == 'full':
                continue
            for size in m['sizes']:
                order = np.random.default_rng(29002 + size).permutation(size)
                gold = np.argsort(order)[:256]
                score = S[:, order]
                for decoder, p in [('row', score.argmax(1)), ('dense', assignment(score))]:
                    pred[f'{learner}_{mode}_{size}_{decoder}'] = p
                    for q, (U, V, keys) in w.items():
                        rows.append(dict(dataset=ds, seed=seed, learner=learner, mode=mode, targets=size, decoder=decoder, query_set=q, **metrics(p, U, V[:, order], gold, q != 'cross_filter')))
    np.savez_compressed(dest.with_suffix('.npz'), **pred)
    dest.write_text(json.dumps(rows, indent=2))
    print('SECONDARY ' + ds + ' ' + str(seed), flush=True)
    del kgs, full, initial, restored, F
    gc.collect()
print('COMPLETE', flush=True)
