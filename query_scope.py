import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import collections
import gc
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, min_weight_full_bipartite_matching
from threadpoolctl import threadpool_limits
R = Path(__file__).resolve().parent
O = R / 'query_scope'
O.mkdir(exist_ok=True)
sys.path.insert(0, str(R / 'deps'))
import faiss
threadpool_limits(2)
faiss.omp_set_num_threads(1)
P = R / 'protocol_query_scope.json'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def dump(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf8')

def partial(S, C):
    n, m = S.shape
    if n == 0:
        return np.empty(0, dtype=np.int64)
    ii = np.repeat(np.arange(n), C.shape[1])
    jj = C.ravel()
    scores = S[ii, jj].astype(np.float64)
    costs = 1 + (scores.max() - scores) / max(float(np.ptp(scores)), 1e-12)
    a = csr_matrix((np.r_[costs, np.full(n, 2 * n + 1.0)], (np.r_[ii, np.arange(n)], np.r_[jj, m + np.arange(n)])), shape=(n, m + n))
    rows, p = min_weight_full_bipartite_matching(a)
    assert np.array_equal(rows, np.arange(n))
    p[p >= m] = -1
    return p

def components(C, m):
    n, k = C.shape
    ii = np.repeat(np.arange(n), k)
    jj = C.ravel() + n
    a = csr_matrix((np.ones(2 * n * k), (np.r_[ii, jj], np.r_[jj, ii])), shape=(n + m, n + m))
    _, labels = connected_components(a, directed=False)
    return labels[:n]

def edge(S, C):
    labels = components(C, S.shape[1])
    p = np.full(len(S), -1, dtype=np.int64)
    for label in np.unique(labels):
        ix = np.flatnonzero(labels == label)
        cols = np.unique(C[ix])
        small = partial(S[np.ix_(ix, cols)], np.searchsorted(cols, C[ix]))
        good = small >= 0
        p[ix[good]] = cols[small[good]]
    return (p, labels)

def pool(S, cols):
    p = np.full(len(S), -1, dtype=np.int64)
    if len(S) and len(cols):
        rows, cc = linear_sum_assignment(S[:, cols], maximize=True)
        p[rows] = cols[cc]
    return p

def answer(p):
    return set(map(int, p[p >= 0]))

def counts(pred, gold):
    a = answer(pred)
    b = answer(gold)
    t = len(a & b)
    return [t, len(a) - t, len(b) - t]

def f1(c):
    t, p, n = np.asarray(c).sum(axis=0)
    return float(2 * t / (2 * t + p + n)) if 2 * t + p + n else 1.0

def top(S, k):
    return np.argsort(-S, axis=1, kind='stable')[:, :k]

def check_solvers():
    rng = np.random.default_rng(53003)
    checked = 0
    for n in [2, 3, 4]:
        for rep in range(24):
            m = n + 2
            S = rng.normal(size=(n, m))
            C = np.array([rng.choice(m, 1 + rep % 3, replace=False) for _ in range(n)])
            p, labels = edge(S, C)
            obj = (int((p >= 0).sum()), float(S[np.arange(n)[p >= 0], p[p >= 0]].sum()))
            best = (-1, -np.inf)
            for q in itertools.product(*[list(c) + [-1] for c in C]):
                real = [t for t in q if t >= 0]
                if len(real) != len(set(real)):
                    continue
                v = (len(real), sum((S[i, t] for i, t in enumerate(q) if t >= 0)))
                best = max(best, v)
            assert obj[0] == best[0] and abs(obj[1] - best[1]) < 1e-10
            checked += 1
    dump(O / 'solver_validation.json', dict(exhaustive_graphs=checked, passed=True))

def evaluate_case(stem, S, C, gold, U, meta, info):
    dest = O / (stem + '.json')
    if dest.exists():
        old = json.loads(dest.read_text())
        assert old['protocol_sha256'] == sha(P.read_bytes())
        return
    n, m = S.shape
    union = np.unique(C)
    ts = time.perf_counter()
    ep, labels = edge(S, C)
    edge_s = time.perf_counter() - ts
    raw = partial(S, C)
    er = ep >= 0
    rr = raw >= 0
    assert er.sum() == rr.sum()
    gap = float(S[np.arange(n)[er], ep[er]].astype(float).sum() - S[np.arange(n)[rr], raw[rr]].astype(float).sum())
    assert abs(gap) < 1e-05
    ts = time.perf_counter()
    pp = pool(S, union)
    pool_s = time.perf_counter() - ts
    rp = C[np.arange(n), S[np.arange(n)[:, None], C].argmax(1)]
    full = {'row': rp, 'edge': ep, 'pool': pp}
    arrays = dict(C=C, gold=gold, U=U, labels=labels, **full)
    plans = {'row_local': 'row', 'edge_local': 'edge', 'edge_closure': 'edge', 'pool_local': 'pool', 'pool_fixed': 'pool'}
    record = {}
    nonempty = U.sum(axis=1) > 0
    closure_sizes = []
    for name, base in plans.items():
        cc = []
        bc = []
        changed = []
        flips = []
        elapsed = []
        predictions = np.full((len(U), n), -2, dtype=np.int32)
        for q, u in enumerate(U):
            ix = np.flatnonzero(u)
            tick = time.perf_counter()
            if not len(ix):
                p = np.empty(0, dtype=np.int64)
            elif name == 'row_local':
                p = C[ix, S[ix[:, None], C[ix]].argmax(1)]
            elif name == 'edge_local':
                p = edge(S[ix], C[ix])[0]
            elif name == 'edge_closure':
                closure = np.flatnonzero(np.isin(labels, labels[ix]))
                p = edge(S[closure], C[closure])[0][np.searchsorted(closure, ix)]
            elif name == 'pool_local':
                p = pool(S[ix], np.unique(C[ix]))
            else:
                p = pool(S[ix], union)
            elapsed.append(time.perf_counter() - tick)
            predictions[q, ix] = p
            baseline = full[base][ix]
            cc.append(counts(p, gold[ix]))
            bc.append(counts(baseline, gold[ix]))
            changed.append(answer(p) != answer(baseline))
            flips.append(int((p != baseline).sum()))
            if name == 'edge_closure':
                closure_sizes.append(int(np.isin(labels, labels[ix]).sum()) if len(ix) else 0)
            if name in ['row_local', 'edge_closure']:
                assert np.array_equal(p, baseline), (stem, name, q)
        arrays[name + '_pred'] = predictions
        arrays[name + '_counts'] = np.array(cc)
        arrays[name + '_baseline_counts'] = np.array(bc)
        arrays[name + '_changed'] = np.array(changed)
        arrays[name + '_flips'] = np.array(flips)
        record[name] = dict(query_f1=f1(cc), baseline_f1=f1(bc), delta_f1=f1(cc) - f1(bc), changed_queries=int(np.sum(changed)), nonempty_queries=int(nonempty.sum()), changed_fraction=float(np.sum(changed) / max(nonempty.sum(), 1)), decision_flips=int(np.sum(flips)), queried_rows=int(U.sum()), decision_flip_fraction=float(np.sum(flips) / max(U.sum(), 1)), decode_s=elapsed, median_nonempty_decode_s=float(np.median(np.asarray(elapsed)[nonempty])) if nonempty.any() else 0.0)
    graph = dict(edge_recall=float((C == gold[:, None]).any(1).mean()), pool_recall=float(np.isin(gold, union).mean()), pool_size=len(union), components=len(np.unique(labels)), largest_component=int(np.bincount(labels).max()), mean_closure_fraction=float(np.mean(np.array(closure_sizes)[nonempty] / n)) if nonempty.any() else 0.0, closure_sizes=closure_sizes, edge_assigned_fraction=float(er.mean()), edge_s=edge_s, pool_s=pool_s, component_global_objective_gap=gap, alternative_optimum_decisions=int((raw != ep).sum()))
    global_metrics = {name: dict(identity_accuracy=float((p == gold).mean()), query_f1=f1([counts(p[u], gold[u]) for u in U]), unsupported_edge_fraction=float(np.mean((p >= 0) & ~(C == p[:, None]).any(1)))) for name, p in full.items()}
    np.savez_compressed(dest.with_suffix('.npz'), **arrays)
    dump(dest, dict(**info, n=n, m=m, queries=meta, plans=record, graph=graph, global_metrics=global_metrics, score_sha256=sha(S.tobytes()), protocol_sha256=sha(P.read_bytes()), producer_sha256=sha(Path(__file__).read_bytes()), score_bytes=S.nbytes))
    print('DONE', stem, 'edge', round(record['edge_local']['changed_fraction'], 3), 'pool', round(record['pool_local']['changed_fraction'], 3), flush=True)

def graph_queries(ds, left, dev):
    from confirmatory_data import facts, b
    rel, val = facts(ds, 1)
    triples = np.loadtxt(b.BASE / ds / 'triples_1', dtype=np.int64)
    devset = set(map(int, dev))
    rels = sorted(rel, key=lambda x: sha(str(x).encode()))[:8]
    two = {}
    byrel = {r: triples[triples[:, 1] == r] for r in rels}
    for r, s in itertools.product(rels, rels):
        two[r, s] = {int(h) for h, _, t in byrel[r] if t in rel[s]}
    U = []
    meta = []
    for family, pred, cap in [('relation', rel, 16), ('value', val, 32), ('twohop', two, 16)]:
        chosen = 0
        seen = set()
        for key in sorted(pred, key=lambda x: sha(json.dumps(x).encode())):
            d = np.array([int(i) in pred[key] for i in dev])
            if not 3 <= d.sum() <= len(dev) / 2 or d.tobytes() in seen:
                continue
            seen.add(d.tobytes())
            U.append(np.array([int(i) in pred[key] for i in left]))
            meta.append(dict(family=family, key=key, development_count=int(d.sum())))
            chosen += 1
            if chosen == cap:
                break
    return (np.array(U), meta)

def density(E, left, right, path):
    if path.exists():
        return np.load(path)
    best = np.full((10, len(right)), -np.inf, dtype=np.float32)
    target = np.asarray(E[right])
    start = time.perf_counter()
    for j in range(0, len(left), 256):
        scores = E[left[j:j + 256]] @ target.T
        v = np.concatenate([best, scores], axis=0)
        best = np.partition(v, len(v) - 10, axis=0)[-10:]
    out = best.mean(axis=0)
    np.save(path, out)
    print('DENSITY', path.name, round(time.perf_counter() - start, 2), flush=True)
    return out

def main():
    from export_frozen import jobs
    from confirmatory_data import b
    check_solvers()
    protocol = json.loads(P.read_text())
    for ds, seed, folder in jobs():
        if ds not in protocol['graph_cohorts']:
            continue
        tag = ds.replace('/', '_')
        last = O / (tag + f'_seed{seed}_block3_hnsw_k64.json')
        if last.exists():
            continue
        man = json.loads((R / 'confirmatory' / (tag + '_identities.json')).read_text())
        start, stop = protocol['graph_cohorts'][ds]
        pairs = np.array(man['test'][start:stop])
        assert pairs.shape == (1024, 2)
        old = set(np.array(man['train'] + man['dev'] + man['test'][:start]).ravel())
        assert not set(pairs.ravel()) & old
        gallery = np.array(man['gallery'])
        lookup = {int(v): j for j, v in enumerate(gallery)}
        fullleft = np.array([int(line.split(b'\t', 1)[0]) for line in (b.BASE / ds / 'ent_ids_1').read_bytes().splitlines() if line], dtype=np.int64)
        prefix = folder + '__' + tag + f'_block{seed}_initial'
        emb = [np.load(R / 'frozen' / (prefix + ('' if d == 0 else f'_depth{d}') + '_node_embeddings.npy'), mmap_mode='r') for d in range(3)]
        dpath = O / (tag + f'_seed{seed}_density.npy')
        den = density(emb[0], fullleft, gallery, dpath)
        index = faiss.IndexHNSWFlat(emb[0].shape[1], 16, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 100
        index.hnsw.efSearch = 64
        t = time.perf_counter()
        index.add(np.ascontiguousarray(emb[0][gallery]))
        build_s = time.perf_counter() - t
        index_bytes = len(faiss.serialize_index(index))
        for block in range(4):
            part = pairs[256 * block:256 * (block + 1)]
            left = part[:, 0]
            gold = np.array([lookup[int(v)] for v in part[:, 1]])
            U, meta = graph_queries(ds, left, np.array(man['dev'])[:, 0])
            dump(O / (tag + f'_block{block}_manifest.json'), dict(dataset=ds, test_offsets=[start + 256 * block, start + 256 * (block + 1)], sources=left.tolist(), targets=gallery.tolist(), gold=gold.tolist(), query_keys=meta, source_disjoint_previous_quality=True, transductive_embeddings=True, protocol_sha256=sha(P.read_bytes())))
            t = time.perf_counter()
            base = emb[0][left] @ emb[0][gallery].T
            S = (base + emb[1][left] @ emb[1][gallery].T + emb[2][left] @ emb[2][gallery].T) / 3
            score_s = time.perf_counter() - t
            t = time.perf_counter()
            cosine = top(base, 64)
            cosine_s = time.perf_counter() - t
            t = time.perf_counter()
            csls = top(2 * base - den[None, :], 64)
            csls_s = time.perf_counter() - t
            t = time.perf_counter()
            _, ann = index.search(np.ascontiguousarray(emb[0][left]), 64)
            ann_s = time.perf_counter() - t
            for retriever, C, search_s in [('cosine', cosine, cosine_s), ('csls10', csls, csls_s), ('hnsw', ann, ann_s)]:
                for k in [1, 4, 16, 64]:
                    name = tag + f'_seed{seed}_block{block}_{retriever}_k{k}'
                    info = dict(dataset=ds, seed=seed, block=block, retriever=retriever, k=k, scorer='DESAlign_depth2', source_scope='full_source_KG_fixed' if retriever == 'csls10' else 'row_local', full_score_s=score_s, retrieval_s=search_s, index_build_s=build_s if retriever == 'hnsw' else None, index_bytes=index_bytes if retriever == 'hnsw' else int(emb[0][gallery].nbytes))
                    evaluate_case(name, S, C[:, :k], gold, U, meta, info)
            gc.collect()
        del emb, index
        gc.collect()
    print('GRAPH COMPLETE', flush=True)
if __name__ == '__main__':
    main()
