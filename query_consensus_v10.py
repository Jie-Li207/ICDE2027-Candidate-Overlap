import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, time, hashlib, itertools
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
threadpool_limits(2)
R = Path(__file__).resolve().parent
O = R / 'consensus_v10'
O.mkdir(exist_ok=True)

def assignment(S):
    return linear_sum_assignment(S, maximize=True)[1]

def answers(p, U):
    out = np.zeros_like(U, dtype=bool)
    out[:, p] = U
    return out

def sinkhorn(S, tau):
    logp = S.astype(np.float64) / tau
    for _ in range(500):
        logp -= logsumexp(logp, axis=1, keepdims=True)
        logp -= logsumexp(logp, axis=0, keepdims=True)
    p = np.exp(logp)
    return (p, float(max(abs(p.sum(0) - 1).max(), abs(p.sum(1) - 1).max())))

def feasible(B, U):
    a, c = np.unique(U.T, axis=0, return_counts=True)
    b, d = np.unique(B.T, axis=0, return_counts=True)
    return bool(np.array_equal(a, b) and np.array_equal(c, d))

def summarize(B, U, G):
    tp = (B & G).sum()
    fp = (B & ~G).sum()
    fn = (~B & G).sum()
    return dict(query_loss=float((fp + fn) / (2 * G.sum())), precision=float(tp / max(B.sum(), 1)), recall=float(tp / G.sum()), joint_feasible=feasible(B, U))

def check():
    rng = np.random.default_rng(101)
    err = []
    for n in [3, 4, 5, 6]:
        perms = list(itertools.permutations(range(n)))
        weights = rng.random(len(perms))
        weights /= weights.sum()
        U = rng.integers(0, 2, (3, n)).astype(bool)
        P = sum((w * np.eye(n)[list(p)] for w, p in zip(weights, perms)))
        q = U.astype(float) @ P
        p = assignment(U.T.astype(float) @ q)
        B = answers(p, U)
        actual = sum((w * np.count_nonzero(B != answers(np.array(g), U)) for w, g in zip(weights, perms)))
        losses = [sum((w * np.count_nonzero(answers(np.array(a), U) != answers(np.array(g), U)) for w, g in zip(weights, perms))) for a in perms]
        err.append(abs(actual - min(losses)))
        assert err[-1] < 1e-10
    return dict(exhaustive_sizes=[3, 4, 5, 6], max_objective_error=max(err))

def main():
    records = []
    for file in sorted((R / 'desalign_v9').glob('block*_scores.npz')):
        z = np.load(file)
        U = z['U'].astype(bool)
        gold = z['gold']
        G = answers(gold, U)
        u = U.astype(float)
        block = int(file.name[5])
        mode = file.name.split('_')[1]
        for depth in ['depth0', 'depth2']:
            S = z[depth]
            t = time.perf_counter()
            p = assignment(S)
            dt = time.perf_counter() - t
            common = dict(block=block, mode=mode, depth=depth, source_signatures=len(np.unique(U.T, axis=0)), n=len(S))
            records.append(dict(**common, method='cosine_assignment', tau=None, posterior_s=0, decision_s=dt, hits=float(np.mean(p == gold)), **summarize(answers(p, U), U, G)))
            for tau in [0.05, 0.01, 0.1]:
                t = time.perf_counter()
                P, res = sinkhorn(S, tau)
                pt = time.perf_counter() - t
                q = u @ P
                for method in ['posterior_assignment', 'consistent_consensus', 'independent_threshold', 'independent_topk', 'rowsoftmax_consensus', 'columnsoftmax_consensus', 'columnsoftmax_topk']:
                    t = time.perf_counter()
                    p = None
                    if method == 'posterior_assignment':
                        p = assignment(P)
                        B = answers(p, U)
                    elif method == 'consistent_consensus':
                        p = assignment(u.T @ q)
                        B = answers(p, U)
                    elif method == 'independent_threshold':
                        B = q >= 0.5
                    elif method == 'independent_topk':
                        B = np.zeros_like(U)
                        for k in range(len(U)):
                            B[k, np.argsort(q[k], kind='stable')[-int(U[k].sum()):]] = True
                    elif method.startswith('columnsoftmax'):
                        cp = np.exp(S / tau - logsumexp(S / tau, axis=0, keepdims=True))
                        cq = u @ cp
                        if method.endswith('consensus'):
                            p = assignment(u.T @ cq)
                            B = answers(p, U)
                        else:
                            B = np.zeros_like(U)
                            for k in range(len(U)):
                                B[k, np.argsort(cq[k], kind='stable')[-int(U[k].sum()):]] = True
                    else:
                        rp = np.exp(S / tau - logsumexp(S / tau, axis=1, keepdims=True))
                        p = assignment(u.T @ (u @ rp))
                        B = answers(p, U)
                    dt = time.perf_counter() - t
                    records.append(dict(**common, method=method, tau=tau, posterior_s=pt, decision_s=dt, marginal_residual=res, hits=None if p is None else float(np.mean(p == gold)), **summarize(B, U, G)))
    (O / 'checks.json').write_text(json.dumps(check(), indent=2))
    (O / 'results.json').write_text(json.dumps(records, indent=2))
    gates = []
    for b in sorted(set((r['block'] for r in records))):
        rows = [r for r in records if r['block'] == b and r['mode'] == 'initial' and (r['depth'] == 'depth2') and (r['tau'] in [None, 0.05])]
        d = {r['method']: r for r in rows}
        base = min((d[x]['query_loss'] for x in ['cosine_assignment', 'posterior_assignment']))
        value = d['consistent_consensus']['query_loss']
        ind = d['independent_topk']['query_loss']
        gain = 1 - value / base
        gates.append(dict(block=b, best_identity_loss=base, consensus_loss=value, independent_loss=ind, relative_gain=gain, passed=bool(gain >= 0.1 and value <= 1.02 * ind)))
    result = dict(protocol_sha256=hashlib.sha256((R / 'protocol_v10_consensus.json').read_bytes()).hexdigest(), gates=gates, records=len(records))
    (O / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
