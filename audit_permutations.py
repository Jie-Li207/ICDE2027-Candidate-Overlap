import json
import numpy as np
from query_scope import R, O, P, sha, dump, edge, pool

def check(stem, S, C, rng):
    records = []
    n, m = S.shape
    original = {'edge': edge(S, C)[0], 'pool': pool(S, np.unique(C))}
    for repeat in range(4):
        rows = rng.permutation(n)
        cols = rng.permutation(m)
        inverse = np.argsort(cols)
        scores = S[np.ix_(rows, cols)]
        candidates = inverse[C[rows]]
        changed = {'edge': edge(scores, candidates)[0], 'pool': pool(scores, np.unique(candidates))}
        for name, permuted in changed.items():
            restored = np.full(n, -1, dtype=np.int64)
            ok = permuted >= 0
            restored[rows[ok]] = cols[permuted[ok]]
            baseline = original[name]
            good = baseline >= 0
            matched = restored >= 0
            a = float(S[np.arange(n)[good], baseline[good]].astype(float).sum())
            b = float(S[np.arange(n)[matched], restored[matched]].astype(float).sum())
            assert good.sum() == matched.sum()
            assert abs(a - b) < 1e-05, (stem, name, a, b)
            records.append(dict(case=stem, repeat=repeat, decoder=name, objective_gap=b - a, alternative_optimum_decisions=int((restored != baseline).sum()), assigned=int(matched.sum())))
    return records

def main():
    from export_frozen import jobs
    rng = np.random.default_rng(53019)
    records = []
    for ds, seed, folder in jobs():
        if seed != 0 or ds not in json.loads(P.read_text())['graph_cohorts']:
            continue
        tag = ds.replace('/', '_')
        mf = json.loads((O / (tag + '_block0_manifest.json')).read_text())
        left = np.array(mf['sources'])
        right = np.array(mf['targets'])
        prefix = folder + '__' + tag + '_block0_initial'
        em = [np.load(R / 'frozen' / (prefix + ('' if d == 0 else f'_depth{d}') + '_node_embeddings.npy'), mmap_mode='r') for d in range(3)]
        S = (em[0][left] @ em[0][right].T + em[1][left] @ em[1][right].T + em[2][left] @ em[2][right].T) / 3
        paths = list(O.glob(tag + '_seed0_block0_*k16.json')) + list(O.glob(tag + '_cdmea_seed0_epoch*_block0_cdi_k16.json'))
        for path in sorted(paths):
            meta = json.loads(path.read_text())
            assert sha(S.tobytes()) == meta['score_sha256']
            with np.load(path.with_suffix('.npz')) as z:
                records.extend(check(path.stem, S, z['C'], rng))
    with np.load(O / 'retail_features.npz') as z:
        E = z['features']
        files = list(z['paths'])
    mf = json.loads((O / 'retail_manifest.json').read_text())
    classes = np.array([p.split('/')[1] for p in files])

    def ids(cs, view):
        return np.array([sorted(np.flatnonzero(classes == c), key=lambda i: files[i])[view] for c in cs])
    X = E[np.r_[ids(mf['fit_classes'], 1), ids(mf['fit_classes'], 2)]].astype(float)
    Y = E[np.r_[ids(mf['fit_classes'], 0), ids(mf['fit_classes'], 0)]].astype(float)
    W = np.eye(X.shape[1]) + X.T @ np.linalg.solve(X @ X.T + np.eye(128), Y - X)
    predicted = (E.astype(float) @ W).astype(np.float32)
    predicted /= np.maximum(np.linalg.norm(predicted, axis=1, keepdims=True), 1e-20)
    for path in sorted(O.glob('Retail_view*_block*_ridge128_cosine_k4.json')):
        meta = json.loads(path.read_text())
        manifest = json.loads((O / f"Retail_view{meta['view']}_block{meta['block']}_manifest.json").read_text())
        S = predicted[manifest['sources']] @ E[manifest['targets']].T
        assert sha(S.tobytes()) == meta['score_sha256']
        with np.load(path.with_suffix('.npz')) as z:
            records.extend(check(path.stem, S, z['C'], rng))
    report = dict(passed=True, cases=len(set((x['case'] for x in records))), permuted_decoder_checks=len(records), maximum_objective_gap=max((abs(x['objective_gap']) for x in records)), alternative_optimum_checks=sum((x['alternative_optimum_decisions'] > 0 for x in records)), records=records, protocol_sha256=sha(P.read_bytes()), scope='Source and target order permutation on seed-zero block-zero graph k16 cases and all ridge/cosine retail k4 blocks; ties are allowed only at equal optimal cardinality and score. Not an independent scientific replication.')
    dump(O / 'permutation_audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2), flush=True)
if __name__ == '__main__':
    main()
