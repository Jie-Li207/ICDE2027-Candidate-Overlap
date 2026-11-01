import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys, json, time, gc
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R / 'deps'))
import numpy as np, faiss
from scipy.special import logsumexp
from tfp_numpy import propagate, norm
from confirmatory_eval import O, metrics
faiss.omp_set_num_threads(2)
OUT = R / 'tfp_fidelity'
OUT.mkdir(exist_ok=True)
BASE = R / 'public_data/desalign/selected/desalign_datasets'

def released_sparse(x, y):
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(np.ascontiguousarray(y))
    s, j = index.search(np.ascontiguousarray(x), 500)
    p = np.exp(s / 0.02).astype(np.float32)
    row_index = np.repeat(np.arange(len(x)), 500)
    col_index = j.ravel()
    for _ in range(15):
        p /= p.sum(1, keepdims=True)
        sums = np.zeros(len(y), np.float32)
        np.add.at(sums, col_index, p.ravel())
        p /= sums[j]
    return (j[np.arange(len(x)), p.argmax(1)], j, p)

def main():
    for ds, folder in [('FBDB15K/norm', 'desalign_gpu'), ('FBYG15K/norm', 'desalign_cpu'), ('DBP15K/zh_en', 'desalign_cpu'), ('DBP15K/fr_en', 'desalign_cpu')]:
        manifest = json.loads((O / (ds.replace('/', '_') + '_identities.json')).read_text())
        train = np.array(manifest['train'])
        z = np.load(O / (ds.replace('/', '_') + '_block0_closed_workload.npz'))
        selected = np.r_[z['left'], z['right']]
        T = np.concatenate([np.loadtxt(BASE / ds / f'triples_{s}', dtype=np.int64) for s in [1, 2]])
        for mode in ['initial', 'full', 'noimage']:
            stem = folder + '__' + ds.replace('/', '_') + '_block0' + ('_noimage' if mode == 'noimage' else '') + '_' + mode
            dest = OUT / (stem + '.json')
            if dest.exists():
                continue
            F = np.load(R / 'frozen' / (stem + '_node_embeddings.npy'), mmap_mode='r')
            out = norm(propagate(F, train, T, selected, seed=24001))
            x = out[:512]
            y = out[512:]
            p, j, P = released_sparse(x, y)
            S = x @ y.T
            lp = S.astype(float) / 0.02
            for _ in range(15):
                lp -= logsumexp(lp, axis=1, keepdims=True)
                lp -= logsumexp(lp, axis=0, keepdims=True)
            dense = np.exp(lp).argmax(1)
            rows = []
            for method, pred in [('released_top500_float32', p), ('common_dense_float64', dense)]:
                for q in ['coarse', 'value', 'cross_filter']:
                    rows.append(dict(dataset=ds, seed=0, block=0, mode=mode, decoder=method, query_set=q, **metrics(pred, z[q + '_U'], z[q + '_V'], z['gold'], q != 'cross_filter')))
            dest.write_text(json.dumps(dict(embedding=stem, changed_row_decisions=int((p != dense).sum()), support_k=500, iterations=15, tau=0.02, rows=rows), indent=2))
            print('FIDELITY ' + stem + ' changed ' + str(int((p != dense).sum())), flush=True)
            del out, F
            gc.collect()
if __name__ == '__main__':
    main()
