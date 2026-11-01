import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, time, gc, collections
import numpy as np
from scipy.sparse import csr_matrix
import desalign_baseline as b
import desalign_crosskg as cross
from tfp_numpy import norm
from export_frozen import jobs
from confirmatory_eval import R, O, RESULT, evaluate, answers
from confirmatory_data import facts
from research_queue import available

def main():
    for ds, seed, _ in jobs():
        stem = ds.replace('/', '_') + f'_seed{seed}'
        complete = O / (stem + '_simple_complete.json')
        if complete.exists():
            continue
        while available() < 10 * 1024 ** 3:
            time.sleep(15)
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
        tick = time.perf_counter()
        for _ in range(8):
            Z = 0.15 * Z0 + 0.85 * (A @ Z)
        Z = norm(Z)
        anchor_s = time.perf_counter() - tick
        restored = initial.copy()
        restored[present] = full[present]
        for mode, F in [('initial', initial), ('full', restored)]:
            F = norm(F)
            for regime in ['closed', 'expanded']:
                for block in range(4):
                    z = np.load(O / (ds.replace('/', '_') + f'_block{block}_{regime}_workload.npz'))
                    left = z['left']
                    right = z['right']
                    tick = time.perf_counter()
                    vis = F[left] @ F[right].T
                    anc = Z[left] @ Z[right].T
                    score_s = time.perf_counter() - tick
                    for learner, S in [('visual_zero_shot', vis), ('anchor128', anc), ('visual_anchor_equal', 0.5 * (vis + anc))]:
                        if learner == 'anchor128' and mode == 'full':
                            continue
                        info = dict(dataset=ds, learner=learner, training_device='cpu', seed=seed, mode=mode, block=block, regime=regime, score='cosine', score_s=score_s, anchor_preprocessing_s=anchor_s, correspondence_labels=0 if learner == 'visual_zero_shot' else 128)
                        evaluate(S, z, info, RESULT / (learner + '__' + stem + f'_{mode}_test{block}_{regime}.json'))
        complete.write_text(json.dumps(dict(dataset=ds, seed=seed, complete=True, restoration='replace originally present features only; retain missing-target imputation')))
        del kgs, full, initial, restored, F
        gc.collect()
if __name__ == '__main__':
    main()
