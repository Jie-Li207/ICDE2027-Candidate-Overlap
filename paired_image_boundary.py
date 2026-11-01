import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, hashlib, collections, gc
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import desalign_baseline as b
import desalign_crosskg as cross
from confirmatory_data import workload
from confirmatory_eval import assignment, metrics
threadpool_limits(2)
R = Path(__file__).resolve().parent
O = R / 'paired_image'
O.mkdir(exist_ok=True)
DS = ['FBDB15K/norm', 'FBYG15K/norm', 'DBP15K/zh_en', 'DBP15K/fr_en']

def sha(x):
    return hashlib.sha256(x).hexdigest()

def main():
    for ds in DS:
        tag = ds.replace('/', '_')
        dest = O / (tag + '_results.json')
        if dest.exists():
            continue
        kgs, full, present, *_ = b.prepare(0) if ds == DS[0] else cross.prepare(ds, 0)
        hashes = {int(i): sha(full[i].tobytes()) for i in np.flatnonzero(present)}
        counts = collections.Counter(hashes.values())
        raw = np.loadtxt(b.BASE / ds / 'ill_ent_ids', dtype=np.int64)
        lc = collections.Counter(raw[:, 0])
        rc = collections.Counter(raw[:, 1])
        man = json.loads((R / 'confirmatory' / (tag + '_identities.json')).read_text())
        excluded = set(np.array(man['train'] + man['dev'] + man['test']).ravel())
        gallery = np.array(man['gallery'])
        pool = []
        for a, c in raw:
            if lc[a] == rc[c] == 1 and a in hashes and (c in hashes) and (hashes[a] == hashes[c]) and (counts[hashes[a]] == 2) and (a not in excluded) and (c not in excluded) and (c in gallery) and np.isfinite(full[a]).all() and (np.linalg.norm(full[a]) > 0):
                pool.append((int(a), int(c)))
        pool.sort(key=lambda p: sha(f'{ds} {p[0]} {p[1]}'.encode()))
        selected = pool[:256]
        record = dict(dataset=ds, safe_shared_image_pairs=len(pool), selected=selected, protocol_sha256=sha((R / 'protocol_paired_image_boundary.json').read_bytes()), seed=0)
        if len(selected) < 64:
            record['status'] = 'insufficient_stratum'
            (O / (tag + '_manifest.json')).write_text(json.dumps(record, indent=2))
            print('SKIP', ds, len(selected), flush=True)
            del kgs, full
            gc.collect()
            continue
        pairs = np.array(selected)
        left = pairs[:, 0]
        others = np.array(sorted(set(gallery) - set(pairs[:, 1])))
        others = others[np.random.default_rng(33001).permutation(len(others))]
        right = np.r_[pairs[:, 1], others]
        n = len(left)
        sizes = sorted(set([n, 4 * n, len(right)]))
        assert not set(pairs.ravel()) & excluded
        w = workload(ds, left, right, right)
        record.update(status='evaluated', sources=left.tolist(), nested_target_order=right.tolist(), sizes=sizes, query_keys={q: v[2] for q, v in w.items()})
        (O / (tag + '_manifest.json')).write_text(json.dumps(record, indent=2))
        arrays = dict(left=left, right=right)
        for q, (U, V, _) in w.items():
            arrays[q + '_U'] = U
            arrays[q + '_V'] = V
        np.savez_compressed(O / (tag + '_workload.npz'), **arrays)
        rows = []
        pred = {}
        folder = 'desalign_gpu' if ds == DS[0] else 'desalign_cpu'

        def evaluate(S, learner, mode):
            for size in sizes:
                order = np.random.default_rng(33002 + size).permutation(size)
                gold = np.argsort(order)[:n]
                score = S[:, order]
                for decoder, p in [('row', score.argmax(1)), ('dense', assignment(score))]:
                    pred[f'{learner}_{mode}_{size}_{decoder}'] = p
                    for q, (U, V, _) in w.items():
                        rows.append(dict(dataset=ds, seed=0, n=n, targets=size, learner=learner, mode=mode, decoder=decoder, query_set=q, **metrics(p, U, V[:, order], gold, q != 'cross_filter')))
        for mode in ['initial', 'full', 'noimage']:
            stem = folder + '__' + tag + '_block0' + ('_noimage' if mode == 'noimage' else '') + '_' + mode
            S = np.zeros((n, len(right)), np.float32)
            for depth in range(3):
                e = np.load(R / 'frozen' / (stem + ('' if depth == 0 else f'_depth{depth}') + '_node_embeddings.npy'), mmap_mode='r')
                S += e[left] @ e[right].T
            evaluate(S / 3, 'DESAlign_depth2', mode)
        initial = kgs['images_list']
        restored = initial.copy()
        restored[present] = full[present]
        for mode, F in [('initial', initial), ('full', restored)]:
            X = F[left]
            Y = F[right]
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
            Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-30)
            evaluate(X @ Y.T, 'visual_cosine', mode)
        dest.write_text(json.dumps(rows, indent=2))
        np.savez_compressed(dest.with_suffix('.npz'), **pred)
        print('BOUNDARY', ds, len(pool), len(rows), flush=True)
        del kgs, full, initial, restored, S
        gc.collect()
    print('COMPLETE', flush=True)
if __name__ == '__main__':
    main()
