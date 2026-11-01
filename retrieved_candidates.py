import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys, json, time, hashlib, gc, collections, itertools
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching, maximum_bipartite_matching
from threadpoolctl import threadpool_limits
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R / 'deps'))
import faiss
from confirmatory_data import facts, predicates, b
from export_frozen import jobs
from tfp_numpy import norm
import desalign_crosskg as cross
threadpool_limits(2)
faiss.omp_set_num_threads(2)
O = R / 'retrieved_candidates'
O.mkdir(exist_ok=True)
KS = [1, 2, 4, 8, 16, 32, 64, 128]
BINS = [0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]

def digest(x):
    return hashlib.sha256(x).hexdigest()

def dump(p, x):
    p.write_text(json.dumps(x, indent=2), encoding='utf8')

def save_workload(ds, cohort, left, right, gold):
    path = O / (ds.replace('/', '_') + '_' + cohort + '_workload.npz')
    if path.exists():
        return (np.load(path), json.loads(path.with_suffix('.json').read_text()))
    U = []
    V = []
    meta = []
    e, val = facts(ds, 1)
    te, _ = facts(ds, 2)

    def optional(pred, ids, limit, mincount):
        try:
            return predicates(pred, ids, limit, mincount)
        except AssertionError:
            return (np.zeros((0, len(ids)), bool), [])
    coarse, ck = optional(e, left, 8, 0.05 * len(left))
    value, vk = optional(val, left, 64, 3)
    _, tk = optional(te, right, 8, 0.05 * len(right))
    target = np.array([[int(i) in te[k] for i in right] for k in tk])
    pairs = [(i, (i + shift) % len(target)) for i in range(len(coarse)) for shift in [0, 3]] if len(target) else []
    original = {'coarse': (coarse, np.ones((len(coarse), len(right)), bool), ck), 'value': (value, np.ones((len(value), len(right)), bool), vk), 'cross_filter': (np.array([coarse[i] for i, j in pairs]), np.array([target[j] for i, j in pairs]), [[ck[i], tk[j]] for i, j in pairs])}
    for family, (u, v, keys) in original.items():
        for a, c, key in zip(u, v, keys):
            U.append(a)
            V.append(c)
            meta.append(dict(family='original_' + family, key=key, bin='original', source_count=int(a.sum())))
    triples = np.loadtxt(b.BASE / ds / 'triples_1', dtype=np.int64)
    rels = sorted(e, key=lambda x: digest(str(x).encode()))[:8]
    two = {}
    for r, s in itertools.product(rels, rels):
        tail = e[s]
        two[r, s] = {int(h) for h, rr, t in triples if rr == r and int(t) in tail}
    for family, pred in [('relation', e), ('relation_object', val), ('two_hop', two)]:
        chosen = collections.Counter()
        seen = set()
        for key in sorted(pred, key=lambda k: digest(json.dumps(k).encode())):
            u = np.array([int(i) in pred[key] for i in left])
            n = int(u.sum())
            if n == 0 or u.tobytes() in seen:
                continue
            bi = int(np.searchsorted(BINS, n / len(left), side='left') - 1)
            assert 0 <= bi < 7
            seen.add(u.tobytes())
            if chosen[bi] >= 8:
                continue
            chosen[bi] += 1
            U.append(u)
            V.append(np.ones(len(right), bool))
            meta.append(dict(family=family, key=key, bin=bi, source_count=n))
    U = np.array(U)
    V = np.array(V)
    np.savez_compressed(path, left=left, right=right, gold=gold, U=U, V=V)
    info = dict(dataset=ds, cohort=cohort, queries=meta, bin_edges=BINS, query_count=len(U), protocol_sha256=digest((R / 'protocol_retrieved_candidates.json').read_bytes()))
    dump(path.with_suffix('.json'), info)
    return (np.load(path), info)

def partial(S, C):
    n, m = S.shape
    k = C.shape[1]
    ii = np.repeat(np.arange(n), k)
    jj = C.ravel()
    v = S[ii, jj].astype(float)
    v = 1 + (v.max() - v) / max(float(np.ptp(v)), 1e-12)
    A = csr_matrix((np.r_[v, np.full(n, 2 * n + 1.0)], (np.r_[ii, np.arange(n)], np.r_[jj, m + np.arange(n)])), shape=(n, m + n))
    a, p = min_weight_full_bipartite_matching(A)
    assert np.array_equal(a, np.arange(n))
    p[p >= m] = -1
    real = p >= 0
    assert len(set(p[real])) == real.sum()
    assert all((p[i] in C[i] for i in np.flatnonzero(real)))
    maxcard = int(np.sum(maximum_bipartite_matching(csr_matrix((np.ones(n * k), (ii, jj)), shape=(n, m)), perm_type='column') >= 0))
    assert real.sum() == maxcard
    return p

def per_query(p, U, V, gold):
    counts = []
    for u, v in zip(U, V):
        g = np.unique(gold[u])
        g = g[v[g]]
        a = np.unique(p[u])
        a = a[a >= 0]
        a = a[v[a]]
        tp = np.intersect1d(a, g, assume_unique=True).size
        counts.append([tp, len(a) - tp, len(g) - tp])
    return np.array(counts, np.int64)

def summarize(p, counts, meta, gold):
    out = []
    groups = collections.defaultdict(list)
    for q, x in enumerate(meta['queries']):
        groups[x['family'], str(x['bin'])].append(q)
    groups['all', 'all'] = list(range(len(counts)))
    real = p >= 0
    for (family, bi), ix in groups.items():
        c = counts[ix]
        tp, fp, fn = c.sum(0)
        den = 2 * c[:, 0] + c[:, 1] + c[:, 2]
        f = np.divide(2 * c[:, 0], den, out=np.ones(len(c), float), where=den > 0)
        out.append(dict(family=family, bin=bi, query_count=len(ix), TP=int(tp), FP=int(fp), FN=int(fn), micro_f1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.0, macro_f1=float(f.mean()), entity_accuracy=float(np.mean(p == gold)), assigned_fraction=float(real.mean()), target_coverage=float(len(set(p[real])) / len(p)), real_collisions=int(real.sum() - len(set(p[real])))))
    return out

def test_solver():
    rng = np.random.default_rng(34001)
    checked = 0
    for n in [2, 3, 4]:
        for rep in range(30):
            m = n + 1
            S = rng.normal(size=(n, m))
            k = 1 + rep % min(3, m)
            C = np.array([rng.choice(m, k, replace=False) for _ in range(n)])
            p = partial(S, C)
            best = (-1, -np.inf)
            for q in itertools.product(*[list(c) + [-1] for c in C]):
                real = [v for v in q if v >= 0]
                if len(real) != len(set(real)):
                    continue
                obj = (len(real), sum((S[i, v] for i, v in enumerate(q) if v >= 0)))
                if obj > best:
                    best = obj
            obj = (int(np.sum(p >= 0)), sum((S[i, v] for i, v in enumerate(p) if v >= 0)))
            assert obj[0] == best[0] and abs(obj[1] - best[1]) < 1e-08
            checked += 1
    dump(O / 'solver_fixtures.json', dict(passed=True, exhaustive_random_graphs=checked))

def run_case(ds, seed, folder, F, cohort, offset, n):
    stem = ds.replace('/', '_') + f'_seed{seed}_{cohort}'
    dest = O / (stem + '_results.json')
    if dest.exists():
        return
    man = json.loads((R / 'confirmatory' / (ds.replace('/', '_') + '_identities.json')).read_text())
    pair = np.array(man['test'][offset:offset + n])
    assert len(pair) == n
    assert not set(pair.ravel()) & set(np.array(man['train'] + man['dev']).ravel())
    if cohort == 'fresh64':
        assert not set(pair.ravel()) & set(np.array(man['test'][:2304]).ravel())
    left = pair[:, 0]
    right = np.array(man['gallery'])
    lookup = {v: i for i, v in enumerate(right)}
    gold = np.array([lookup[v] for v in pair[:, 1]])
    z, wm = save_workload(ds, cohort, left, right, gold)
    U = z['U']
    V = z['V']
    prefix = folder + '__' + ds.replace('/', '_') + f'_block{seed}_initial'
    emb = [np.load(R / 'frozen' / (prefix + ('' if d == 0 else f'_depth{d}') + '_node_embeddings.npy'), mmap_mode='r') for d in range(3)]
    tick = time.perf_counter()
    S = sum((e[left] @ e[right].T for e in emb)) / 3
    score_s = time.perf_counter() - tick
    rows = []
    arrays = {}
    conditions = []

    def record(cid, p, info):
        c = per_query(p, U, V, gold)
        arrays[cid + '_p'] = p
        arrays[cid + '_counts'] = c
        conditions.append(dict(id=cid, **info))
        rows.extend((dict(id=cid, dataset=ds, seed=seed, cohort=cohort, **info, **x) for x in summarize(p, c, wm, gold)))
    for scheme, cols in [('full', np.arange(len(right))), ('oracle_closed', gold)]:
        A = S[:, cols]
        tick = time.perf_counter()
        p = cols[A.argmax(1)]
        record(scheme + '_row', p, dict(retriever=scheme, k=len(cols), decoder='row', decode_s=time.perf_counter() - tick))
        tick = time.perf_counter()
        p = cols[linear_sum_assignment(A, maximize=True)[1]]
        record(scheme + '_dense', p, dict(retriever=scheme, k=len(cols), decoder='dense', decode_s=time.perf_counter() - tick))
    for retriever, E in [('multimodal_depth0', emb[0]), ('visual_zero_label', F)]:
        tick = time.perf_counter()
        index = faiss.IndexFlatIP(E.shape[1])
        index.add(np.ascontiguousarray(E[right], dtype=np.float32))
        build = time.perf_counter() - tick
        for k in KS:
            tick = time.perf_counter()
            dist, C = index.search(np.ascontiguousarray(E[left], dtype=np.float32), k)
            search = time.perf_counter() - tick
            assert (C >= 0).all()
            union = np.unique(C)
            recall = float(np.mean((C == gold[:, None]).any(1)))
            indeg = np.bincount(C.ravel(), minlength=len(right))
            info = dict(retriever=retriever, k=k, edge_recall=recall, edge_precision=recall / k, union_target_recall=float(np.isin(gold, union).mean()), union_size=len(union), max_target_indegree=int(indeg.max()), index_build_s=build, index_search_s=search, index_vector_bytes=int(len(right) * E.shape[1] * 4), full_score_s=score_s)
            arrays[retriever + f'_k{k}_edges'] = C
            tick = time.perf_counter()
            p = C[np.arange(n), S[np.arange(n)[:, None], C].argmax(1)]
            record(retriever + f'_k{k}_row', p, dict(**info, decoder='row', decode_s=time.perf_counter() - tick))
            tick = time.perf_counter()
            p = partial(S, C)
            record(retriever + f'_k{k}_edge', p, dict(**info, decoder='edge', decode_s=time.perf_counter() - tick))
            tick = time.perf_counter()
            p = np.full(n, -1, int)
            rr, cc = linear_sum_assignment(S[:, union], maximize=True)
            p[rr] = union[cc]
            unsupported = float(np.mean((p >= 0) & ~(C == p[:, None]).any(1)))
            record(retriever + f'_k{k}_union', p, dict(**info, decoder='union', unsupported_edge_fraction=unsupported, decode_s=time.perf_counter() - tick))
        del index
    np.savez_compressed(dest.with_suffix('.npz'), **arrays)
    dump(dest, dict(dataset=ds, seed=seed, cohort=cohort, score_sha256=digest(S.tobytes()), protocol_sha256=digest((R / 'protocol_retrieved_candidates.json').read_bytes()), producer_sha256=digest(Path(__file__).read_bytes()), conditions=conditions, rows=rows))
    print('DONE ' + stem + ' ' + str(len(rows)), flush=True)
    del emb, S
    gc.collect()

def main():
    test_solver()
    for ds, seed, folder in jobs():
        if all(((O / (ds.replace('/', '_') + f'_seed{seed}_{c}_results.json')).exists() for c in ['diagnostic256', 'fresh64'])):
            continue
        kgs, full, present, *_ = b.prepare(seed) if ds == 'FBDB15K/norm' else cross.prepare(ds, seed)
        F = norm(kgs['images_list'])
        del kgs, full, present
        gc.collect()
        for cohort, offset, n in [('diagnostic256', 2048, 256), ('fresh64', 2304, 64)]:
            run_case(ds, seed, folder, F, cohort, offset, n)
        del F
        gc.collect()
    print('ALL COMPLETE', flush=True)
if __name__ == '__main__':
    main()
