import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, time, gc, argparse
from pathlib import Path
import numpy as np
from tfp_numpy import propagate, norm
from confirmatory_eval_v27 import R, O, RESULT, evaluate
BASE = R / 'public_data/desalign/selected/desalign_datasets'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    opt = parser.parse_args()
    done = 0
    rng = np.random.default_rng(27003)
    x = norm(rng.normal(size=(31, 13)).astype(np.float32))
    T = np.c_[rng.integers(31, size=91), rng.integers(5, size=91), rng.integers(31, size=91)]
    train = np.array([[0, 30], [1, 29]])
    sel = np.arange(28)
    a = propagate(x, train, T, sel, rel_dim=11, mini_dim=4)
    b = propagate(x, train, T, sel, rel_dim=11, mini_dim=4, project_first=True)
    err = float(abs(a - b).max())
    assert err < 1e-05
    (O / 'tfp_associativity_check.json').write_text(json.dumps(dict(max_error=err, selection='project-first for full real target gallery, algebraically equivalent; no quality selection')))
    for audit in sorted((R / 'frozen_v27').glob('*_audit.json')):
        meta = json.loads(audit.read_text())
        stem = audit.name.removesuffix('_audit.json')
        ds = meta['dataset']
        seed = meta['seed']
        man = json.loads((O / (ds.replace('/', '_') + '_identities.json')).read_text())
        train = np.array(man['train'])
        allleft = np.array(man['test'][:2048])[:, 0]
        gallery = np.array(man['gallery'])
        selected = np.r_[allleft, gallery]
        lookup = {int(i): j for j, i in enumerate(selected)}
        T = np.concatenate([np.loadtxt(BASE / ds / f'triples_{side}', dtype=np.int64) for side in [1, 2]])
        for mode in ['noimage'] if meta['noimage'] else ['initial', 'full']:
            prefix = 'TFP__' + stem + '_' + mode
            if all(((RESULT / (prefix + f'_test{block}_{regime}.json')).exists() for regime in ['closed', 'expanded'] for block in range(4))):
                continue
            F = np.load(R / 'frozen_v27' / (stem + f'_{mode}_node_embeddings.npy'), mmap_mode='r')
            tick = time.perf_counter()
            out = norm(propagate(F, train, T, selected, rel_dim=512, mini_dim=16, seed=24001 + seed, project_first=True))
            prep_s = time.perf_counter() - tick
            for regime in ['closed', 'expanded']:
                for block in range(4):
                    z = np.load(O / (ds.replace('/', '_') + f'_block{block}_{regime}_workload.npz'))
                    li = np.array([lookup[int(i)] for i in z['left']])
                    ri = np.array([lookup[int(i)] for i in z['right']])
                    tick = time.perf_counter()
                    S = out[li] @ out[ri].T
                    info = dict(dataset=ds, learner='DESAlign+TFP', training_device='gpu' if 'gpu' in meta['folder'] else 'cpu', seed=seed, mode=mode, block=block, regime=regime, score='tfp_cosine', score_s=time.perf_counter() - tick, tfp_preprocessing_s=prep_s, checkpoint_sha256=meta['checkpoint_sha256'])
                    evaluate(S, z, info, RESULT / (prefix + f'_test{block}_{regime}.json'))
            del out, F
            gc.collect()
            done += 1
            if opt.limit and done >= opt.limit:
                return
if __name__ == '__main__':
    main()
