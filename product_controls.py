import csv, json, re, collections, time, gc, io, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from retrieved_candidates import R, KS, BINS, digest, dump, partial, per_query, summarize
CLEAN = '--exclude-shared-text' in sys.argv
O = R / ('product_text_boundary' if CLEAN else 'product_controls')
O.mkdir(exist_ok=True)

def read(p):
    try:
        return list(csv.DictReader(io.StringIO(p.read_text(encoding='utf-8-sig'))))
    except UnicodeDecodeError:
        return list(csv.DictReader(io.StringIO(p.read_text(encoding='latin1'))))

def text(row):
    return ' '.join((row.get(k, '') for k in ['name', 'title', 'description', 'manufacturer']))

def workload(A, left, right, gold, stem):
    pred = collections.defaultdict(set)
    prices = []
    for i, row in enumerate(A):
        for token in set(re.findall('[a-z]{3,}', (row.get('name', '') + ' ' + row.get('title', '')).lower())):
            pred['token', token].add(i)
        maker = row.get('manufacturer', '').strip().lower()
        if maker:
            pred['manufacturer', maker].add(i)
        try:
            price = float(re.sub('[,$]', '', row.get('price', '')))
            if np.isfinite(price) and price > 0:
                prices.append((i, price))
        except ValueError:
            pass
    if prices:
        for q in np.arange(0.1, 1.0, 0.1):
            threshold = float(np.quantile([p for i, p in prices], q))
            pred['price_le', threshold] = {i for i, p in prices if p <= threshold}
    U = []
    meta = []
    counts = collections.Counter()
    seen = collections.defaultdict(set)
    for key in sorted(pred, key=lambda x: digest(json.dumps(x).encode())):
        u = np.array([int(i) in pred[key] for i in left])
        n = int(u.sum())
        if not n or u.tobytes() in seen[key[0]]:
            continue
        bi = int(np.searchsorted(BINS, n / len(left), side='left') - 1)
        seen[key[0]].add(u.tobytes())
        if counts[key[0], bi] >= 8:
            continue
        counts[key[0], bi] += 1
        U.append(u)
        meta.append(dict(family=key[0], bin=bi, key=key[1], source_count=n))
    U = np.array(U)
    V = np.ones((len(U), len(right)), bool)
    wm = dict(queries=meta, bin_edges=BINS, query_count=len(U))
    np.savez_compressed(O / (stem + '_workload.npz'), U=U, V=V, left=left, right=right, gold=gold)
    dump(O / (stem + '_workload.json'), wm)
    return (U, V, wm)

def main():
    for name, a, b in [('Abt-Buy', 'Abt.csv', 'Buy.csv'), ('Amazon-GoogleProducts', 'Amazon.csv', 'GoogleProducts.csv')]:
        if (O / (name + '_complete.json')).exists():
            continue
        src = R / 'public_data/products' / name
        A = read(src / a)
        B = read(src / b)
        ai = {r['id']: i for i, r in enumerate(A)}
        bi = {r['id']: i for i, r in enumerate(B)}
        raw = read(next(src.glob('*perfectMapping.csv')))
        keys = list(raw[0])
        pairs = [(ai[r[keys[0]]], bi[r[keys[1]]]) for r in raw]
        lc = collections.Counter((x for x, y in pairs))
        rc = collections.Counter((y for x, y in pairs))
        eligible = [p for p in pairs if lc[p[0]] == rc[p[1]] == 1]
        eligible.sort(key=lambda p: digest((name + '|' + A[p[0]]['id'] + '|' + B[p[1]]['id']).encode()))
        eligible = np.array(eligible)
        train = eligible[:128]
        dev = eligible[128:256]
        test = eligible[256:768]
        assert len(train) == 128 and len(test) > 256
        textsA = [' '.join(text(x).lower().split()) for x in A]
        textsB = [' '.join(text(x).lower().split()) for x in B]
        knownA = {textsA[i] for i in eligible[:256, 0] if textsA[i]}
        knownB = {textsB[i] for i in eligible[:256, 1] if textsB[i]}
        right = np.array(sorted(set(range(len(B))) - set(eligible[:256, 1])))
        lookup = {v: i for i, v in enumerate(right)}
        dump(O / (name + '_identities.json'), dict(dataset=name, left_n=len(A), right_n=len(B), raw_pairs=len(pairs), eligible_pairs=len(eligible), excluded_nonbijective=len(pairs) - len(eligible), train=train.tolist(), dev=dev.tolist(), test=test.tolist(), gallery=right.tolist(), protocol_sha256=digest((R / 'protocol_products.json').read_bytes())))
        tick = time.perf_counter()
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=50000, dtype=np.float32)
        X = vec.fit_transform([text(r) for r in A + B])
        build_s = time.perf_counter() - tick
        svd = TruncatedSVD(n_components=128, random_state=35001)
        E = normalize(svd.fit_transform(X))
        LA = E[:len(A)]
        RB = E[len(A):]
        W = np.linalg.solve(LA[train[:, 0]].T @ LA[train[:, 0]] + np.eye(128), LA[train[:, 0]].T @ RB[train[:, 1]])
        mapped = normalize(LA @ W)
        for block, part in enumerate([test[:256], test[256:]]):
            if CLEAN:
                part = np.array([p for p in part if textsA[p[0]] not in knownA and textsB[p[1]] not in knownB])
            if not len(part):
                continue
            left = part[:, 0]
            gold = np.array([lookup[x] for x in part[:, 1]])
            n = len(left)
            stem = name + f'_block{block}'
            U, V, wm = workload(A, left, right, gold, stem)
            tick = time.perf_counter()
            retrieval = (X[left] @ X[len(A) + right].T).toarray()
            retrieval_s = time.perf_counter() - tick
            Cfull = np.argsort(-retrieval, axis=1, kind='stable')[:, :128]
            for scorer, S in [('tfidf_zero_label', retrieval), ('ridge128', mapped[left] @ RB[right].T)]:
                arrays = {}
                rows = []
                conditions = []

                def record(cid, p, info):
                    c = per_query(p, U, V, gold)
                    arrays[cid + '_p'] = p
                    arrays[cid + '_counts'] = c
                    conditions.append(dict(id=cid, **info))
                    rows.extend((dict(id=cid, dataset=name, scorer=scorer, block=block, **info, **x) for x in summarize(p, c, wm, gold)))
                for scheme, cols in [('full', np.arange(len(right))), ('oracle_closed', gold)]:
                    Q = S[:, cols]
                    record(scheme + '_row', cols[Q.argmax(1)], dict(retriever=scheme, k=len(cols), decoder='row'))
                    rr, pp = linear_sum_assignment(Q, maximize=True)
                    assert len(rr) == n
                    record(scheme + '_dense', cols[pp], dict(retriever=scheme, k=len(cols), decoder='dense'))
                for k in KS:
                    C = Cfull[:, :k]
                    arrays[f'tfidf_k{k}_edges'] = C
                    union = np.unique(C)
                    recall = float(np.mean((C == gold[:, None]).any(1)))
                    info = dict(retriever='tfidf', k=k, edge_recall=recall, edge_precision=recall / k, union_target_recall=float(np.isin(gold, union).mean()), union_size=len(union), tfidf_build_s=build_s, retrieval_s=retrieval_s)
                    for decoder in ['row', 'edge', 'union']:
                        tick = time.perf_counter()
                        if decoder == 'row':
                            p = C[np.arange(n), S[np.arange(n)[:, None], C].argmax(1)]
                        elif decoder == 'edge':
                            p = partial(S, C)
                        else:
                            p = np.full(n, -1, int)
                            rr, cc = linear_sum_assignment(S[:, union], maximize=True)
                            p[rr] = union[cc]
                        extra = dict(unsupported_edge_fraction=float(np.mean((p >= 0) & ~(C == p[:, None]).any(1)))) if decoder == 'union' else {}
                        record(f'tfidf_k{k}_{decoder}', p, dict(**info, decoder=decoder, decode_s=time.perf_counter() - tick, **extra))
                dest = O / (stem + '_' + scorer + '_results.json')
                np.savez_compressed(dest.with_suffix('.npz'), **arrays)
                dump(dest, dict(conditions=conditions, rows=rows, score_sha256=digest(S.tobytes()), protocol_sha256=digest((R / 'protocol_products.json').read_bytes()), producer_sha256=digest(Path(__file__).read_bytes())))
                print('DONE ' + dest.stem, flush=True)
        dump(O / (name + '_complete.json'), dict(complete=True, feature_count=X.shape[1], fit_labels=128, development_tuning=False))
        del X, E, LA, RB, W, mapped
        gc.collect()
if __name__ == '__main__':
    main()
