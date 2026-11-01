import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, time, hashlib, collections, argparse, gc
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from threadpoolctl import threadpool_limits
from posterior_audit import rounded, residual
threadpool_limits(2)
R = Path(__file__).resolve().parent
O = R / 'confirmatory'
RESULT = O / 'results'
RESULT.mkdir(exist_ok=True)

def assignment(S):
    return linear_sum_assignment(S, maximize=True)[1]

def stable(S, verify=True):
    n, m = S.shape
    pref = np.argsort(-S, axis=1, kind='stable')
    nxt = np.zeros(n, int)
    owner = np.full(m, -1)
    p = np.full(n, -1)
    free = collections.deque(range(n))
    while free:
        i = free.popleft()
        j = pref[i, nxt[i]]
        nxt[i] += 1
        old = owner[j]
        if old < 0 or S[i, j] > S[old, j] or (S[i, j] == S[old, j] and i < old):
            p[i] = j
            owner[j] = i
            if old >= 0:
                free.append(int(old))
        else:
            free.append(i)
    if verify:
        assert len(np.unique(p)) == n
        current = np.full(m, -np.inf)
        owned = owner >= 0
        current[owned] = S[owner[owned], np.flatnonzero(owned)]
        assert not ((S > S[np.arange(n), p, None]) & (S > current[None, :])).any()
    return p

def sparse(S):
    n, m = S.shape
    k = min(16, m)
    attempts = []
    while True:
        js = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ii = np.repeat(np.arange(n), k)
        jj = js.ravel()
        cost = S.max() - S[ii, jj] + 1.0
        A = csr_matrix((cost, (ii, jj)), shape=S.shape)
        attempts.append(k)
        try:
            rr, p = min_weight_full_bipartite_matching(A)
            assert np.array_equal(rr, np.arange(n))
            return (p, attempts)
        except ValueError:
            if k == m:
                raise
            k = min(2 * k, m)

def answers(p, U, m):
    B = np.zeros((len(U), m), bool)
    for q, u in enumerate(U):
        B[q, np.unique(p[u])] = True
    return B

def metrics(p, U, V, gold, unary=True):
    m = V.shape[1]
    B = answers(p, U, m) & V
    G = answers(gold, U, m) & V
    K = G.sum(1)
    M = B.sum(1)
    TP = (B & G).sum(1)
    FP = (B & ~G).sum(1)
    FN = (~B & G).sum(1)
    den = K + M
    f1 = np.divide(2 * TP, den, out=np.ones(len(U), float), where=den > 0)
    if unary:
        assert np.array_equal(FP + FN, K - M + 2 * FP)
        assert np.all(K >= M)
    return dict(entity_accuracy=float(np.mean(p == gold)), target_coverage=float(len(np.unique(p)) / len(p)), key_collisions=len(p) - len(np.unique(p)), injective=bool(len(np.unique(p)) == len(p)), query_count=len(U), empty_gold_queries=int(np.sum(K == 0)), micro_f1=float(2 * TP.sum() / den.sum()) if den.sum() else 1.0, macro_f1=float(f1.mean()), precision=float(TP.sum() / M.sum()) if M.sum() else float(K.sum() == 0), recall=float(TP.sum() / K.sum()) if K.sum() else 1.0, query_loss=float((FP + FN).sum() / max(2 * K.sum(), 1)), coverage_deficit_term=float((K - M).sum() / max(2 * K.sum(), 1)) if unary else None, false_positive_term=float(FP.sum() / max(K.sum(), 1)) if unary else None, TP=int(TP.sum()), FP=int(FP.sum()), FN=int(FN.sum()), gold_answers=int(K.sum()), predicted_answers=int(M.sum()), per_query_f1=f1.tolist())

def decoders(S, closed):
    t = time.perf_counter()
    p = S.argmax(1)
    yield ('row', p, dict(decoder_s=time.perf_counter() - t))
    t = time.perf_counter()
    p = assignment(S)
    raw = p
    yield ('dense', p, dict(decoder_s=time.perf_counter() - t))
    t = time.perf_counter()
    p = stable(S)
    yield ('stable', p, dict(decoder_s=time.perf_counter() - t))
    t = time.perf_counter()
    p, ks = sparse(S)
    gap = float(np.sum(S[np.arange(len(S)), raw].astype(float) - S[np.arange(len(S)), p].astype(float)))
    assert gap >= -0.0001
    yield ('sparse', p, dict(decoder_s=time.perf_counter() - t, sparse_k_attempts=ks, raw_objective_regret=gap))
    if not closed:
        return
    logp = S.astype(float) / 0.02
    norm_s = 0.0
    for it in range(1, 501):
        t = time.perf_counter()
        logp -= logsumexp(logp, axis=1, keepdims=True)
        logp -= logsumexp(logp, axis=0, keepdims=True)
        norm_s += time.perf_counter() - t
        if it not in [15, 500]:
            continue
        t = time.perf_counter()
        P = np.exp(logp)
        unrepaired = residual(P)
        if it == 500:
            P = rounded(P)
        norm_s += time.perf_counter() - t
        order = np.sort(P, axis=1)
        gap = order[:, -1] - order[:, -2]
        info = dict(normalize_s=norm_s, marginal_residual=residual(P), unrepaired_residual=unrepaired, near_tie_rows=int(np.sum(gap < 1e-10)), min_top2_gap=float(gap.min()))
        tt = time.perf_counter()
        p = P.argmax(1)
        yield (f'ot{it}_row', p, dict(**info, decoder_s=time.perf_counter() - tt))
        tt = time.perf_counter()
        p = assignment(P)
        yield (f'ot{it}_dense', p, dict(**info, decoder_s=time.perf_counter() - tt))
    plog = assignment(logp)
    objective_gap = float(np.sum(S[np.arange(len(S)), raw].astype(float) - S[np.arange(len(S)), plog].astype(float)))
    assert abs(objective_gap) < 0.0001

def evaluate(S, z, info, dest):
    if dest.exists():
        cached = json.loads(dest.read_text())
        assert cached['score_sha256'] == hashlib.sha256(S.tobytes()).hexdigest(), 'Score input changed; preserve old outcomes and evaluate in a fresh result directory'
        return cached['rows']
    rows = []
    pred = {}
    closed = info['regime'] == 'closed'
    for method, p, timing in decoders(S, closed):
        assert p.min() >= 0 and p.max() < S.shape[1]
        pred[method] = p
        for q in ['coarse', 'value', 'cross_filter']:
            U = z[q + '_U']
            V = z[q + '_V']
            rows.append(dict(**info, decoder=method, query_set=q, source_n=S.shape[0], target_n=S.shape[1], **timing, **metrics(p, U, V, z['gold'], q != 'cross_filter')))
    np.savez_compressed(dest.with_suffix('.npz'), **pred)
    dest.write_text(json.dumps(dict(score_sha256=hashlib.sha256(S.tobytes()).hexdigest(), rows=rows), indent=2))
    print('EVALUATED ' + dest.stem, flush=True)
    return rows

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--regime', choices=['closed', 'expanded', 'both'], default='both')
    opt = a.parse_args()
    for audit in sorted((R / 'frozen').glob('*_audit.json')):
        meta = json.loads(audit.read_text())
        stem = audit.name.removesuffix('_audit.json')
        ds = meta['dataset']
        seed = meta['seed']
        modes = ['noimage'] if meta['noimage'] else ['initial', 'full']
        for mode in modes:
            emb = [np.load(R / 'frozen' / (stem + f'_{mode}' + ('' if d == 0 else f'_depth{d}') + '_node_embeddings.npy'), mmap_mode='r') for d in range(3)]
            for regime in ['closed', 'expanded'] if opt.regime == 'both' else [opt.regime]:
                for block in range(4):
                    z = np.load(O / (ds.replace('/', '_') + f'_block{block}_{regime}_workload.npz'))
                    left = z['left']
                    right = z['right']
                    S = None
                    tick = time.perf_counter()
                    for d in range(3):
                        if d == 0:
                            S = emb[d][left] @ emb[d][right].T
                            base = S.copy()
                        else:
                            S += emb[d][left] @ emb[d][right].T
                    S /= 3
                    score_s = time.perf_counter() - tick
                    for depth, scores in [('depth0', base), ('depth2', S)]:
                        dest = RESULT / (stem + f'_{mode}_test{block}_{regime}_{depth}.json')
                        info = dict(dataset=ds, learner='DESAlign', training_device='gpu' if 'gpu' in meta['folder'] else 'cpu', seed=seed, mode=mode, block=block, regime=regime, score=depth, score_s=score_s, checkpoint_sha256=meta['checkpoint_sha256'])
                        evaluate(scores, z, info, dest)
                    del S, base
                    gc.collect()
            del emb
            gc.collect()
    allrows = []
    for f in sorted(RESULT.glob('*.json')):
        allrows.extend(json.loads(f.read_text())['rows'])
    (O / 'all_results.json').write_text(json.dumps(allrows))
    print('COMPLETE ' + str(len(allrows)), flush=True)
if __name__ == '__main__':
    main()
