import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, gc, time, collections
import numpy as np
from scipy.sparse import csr_matrix
import desalign_baseline as b
import desalign_crosskg as cross
from confirmatory_data import facts
from export_frozen import jobs
from confirmatory_eval import R, O, answers
from tfp_numpy import norm
from research_queue import available
OUT = O / 'direct'
OUT.mkdir(exist_ok=True)

def feasible(B, U):
    count = collections.Counter(map(bytes, np.packbits(U.T, axis=1)))
    zero = bytes(np.zeros((U.shape[0] + 7) // 8, np.uint8))
    count[zero] += B.shape[1] - U.shape[1]
    return count == collections.Counter(map(bytes, np.packbits(B.T, axis=1)))

def quality(B, G):
    K = G.sum(1)
    M = B.sum(1)
    TP = (B & G).sum(1)
    den = K + M
    return dict(micro_f1=float(2 * TP.sum() / den.sum()) if den.sum() else 1.0, macro_f1=float(np.divide(2 * TP, den, out=np.ones(len(K), float), where=den > 0).mean()), TP=int(TP.sum()), FP=int((B & ~G).sum()), FN=int((~B & G).sum()), query_count=len(K), empty_gold_queries=int((K == 0).sum()))

def main():
    for ds, seed, _ in jobs():
        dest = OUT / (ds.replace('/', '_') + f'_seed{seed}.json')
        if dest.exists():
            continue
        while available() < 10 * 1024 ** 3:
            time.sleep(15)
        kgs, full, present, *_ = b.prepare(seed) if ds == 'FBDB15K/norm' else cross.prepare(ds, seed)
        n = kgs['ent_num']
        train = kgs['train_ill']
        F = norm(kgs['images_list'].astype(np.float64))
        x = F[train[:, 1]]
        T = np.concatenate([np.loadtxt(b.BASE / ds / f'triples_{s}', dtype=np.int64) for s in [1, 2]])
        edges = np.r_[T[:, [0, 2]], T[:, [2, 0]]]
        A = csr_matrix((np.ones(len(edges), np.float32), (edges[:, 0], edges[:, 1])), shape=(n, n))
        A.sum_duplicates()
        A.data[:] = 1
        A = A.multiply(1 / np.maximum(np.asarray(A.sum(1)), 1)).tocsr()
        exist, value = facts(ds, 1)
        rows = []
        for block in range(4):
            for regime in ['closed', 'expanded']:
                prefix = ds.replace('/', '_') + f'_block{block}_{regime}'
                z = np.load(O / (prefix + '_workload.npz'))
                keys = json.loads((O / (prefix + '_queries.json')).read_text())
                right = z['right']
                predictions = {}
                for family in ['coarse', 'value']:
                    members = exist if family == 'coarse' else value
                    Y = np.array([[int(a) in members[k if family == 'coarse' else tuple(k)] for k in keys[family]] for a in train[:, 0]], float)
                    mean = Y.mean(0)
                    coef = np.linalg.solve(x @ x.T + np.eye(128), Y - mean)
                    ridge = np.clip(mean + F[right] @ x.T @ coef, 0, 1).T
                    start = np.zeros((n, len(mean) + 1), np.float32)
                    start[train[:, 1], :-1] = Y
                    start[train[:, 1], -1] = 1
                    prop = start.copy()
                    for _ in range(8):
                        prop = 0.15 * start + 0.85 * (A @ prop)
                    den = prop[right, -1]
                    graph = np.divide(prop[right, :-1], den[:, None], out=np.broadcast_to(mean, (len(right), len(mean))).copy(), where=den[:, None] > 0).T
                    for model, q in [('target_visual_ridge', ridge), ('target_graph_propagation', graph)]:
                        for rule in ['threshold', 'source_cardinality_topk']:
                            if rule == 'threshold':
                                B = q >= 0.5
                            else:
                                B = np.zeros_like(q, bool)
                                for j, u in enumerate(z[family + '_U']):
                                    k = int(u.sum())
                                    if k:
                                        B[j, np.argsort(-q[j], kind='stable')[:k]] = True
                            G = answers(z['gold'], z[family + '_U'], len(right))
                            rows.append(dict(dataset=ds, seed=seed, block=block, regime=regime, learner=model, decision=rule, query_set=family, labels=128, joint_query_feasible=feasible(B, z[family + '_U']), **quality(B, G)))
                            predictions[family, model, rule] = B
                            if family == 'coarse':
                                indices = []
                                for sourcekey, targetkey in keys['cross_filter']:
                                    indices.append(keys['coarse'].index(sourcekey))
                                BB = B[indices] & z['cross_filter_V']
                                GG = answers(z['gold'], z['cross_filter_U'], len(right)) & z['cross_filter_V']
                                rows.append(dict(dataset=ds, seed=seed, block=block, regime=regime, learner=model, decision=rule, query_set='cross_filter', labels=128, joint_query_feasible=None, **quality(BB, GG)))
                for model in ['target_visual_ridge', 'target_graph_propagation']:
                    for rule in ['threshold', 'source_cardinality_topk']:
                        U = np.r_[z['coarse_U'], z['value_U']]
                        B = np.r_[predictions['coarse', model, rule], predictions['value', model, rule]]
                        rows.append(dict(dataset=ds, seed=seed, block=block, regime=regime, learner=model, decision=rule, query_set='coarse_plus_value', labels=128, joint_query_feasible=feasible(B, U), **quality(B, answers(z['gold'], U, len(right)))))
        dest.write_text(json.dumps(rows, indent=2))
        print('DIRECT ' + ds + ' ' + str(seed), flush=True)
        del kgs, F, full
        gc.collect()
if __name__ == '__main__':
    main()
