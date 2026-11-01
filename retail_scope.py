import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import collections
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from query_scope import R, O, P, sha, dump, top, evaluate_case

def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-20)

def truth_value(x):
    return str(x).strip().lower() == 'yes'

def build_queries(dev, source):
    rows = []
    meta = []
    for value in [1, 2, 3, 5, 10]:
        rows.append(np.array(source['price'] <= value))
        meta.append(dict(family='price', key=value))
    rows.append(np.array(source['relative_discount'].notna() | source['absolute_discount'].notna()))
    meta.append(dict(family='discount', key='not_missing'))
    rows.append(np.array([truth_value(v) for v in source['different_sorts']]))
    meta.append(dict(family='different_sorts', key='yes'))
    for field in ['brand', 'product_category']:
        values = collections.Counter((str(x) for x in dev[field].dropna()))
        keys = sorted([k for k, v in values.items() if v >= 2], key=lambda x: sha(x.encode()))[:8]
        for key in keys:
            rows.append(np.array([str(x) == key for x in source[field]]))
            meta.append(dict(family=field, key=key))
    return (np.array(rows), meta)

def main():
    D = R / 'public_data' / 'retail'
    z = np.load(O / 'retail_features.npz')
    files = list(z['paths'])
    df = pd.read_parquet(D / 'test.parquet')
    df['path'] = ['test/' + str(a) + '/' + str(b) for a, b in zip(df['class'], df['filename'])]
    df = df.set_index('path').loc[files].reset_index()
    classes = np.array(df['class'].astype(str))
    pixel = z['pixel_sha256']
    groups = collections.defaultdict(list)
    for i, h in enumerate(pixel):
        groups[str(h)].append(i)
    duplicated = {str(classes[i]) for ix in groups.values() if len(ix) > 1 for i in ix}
    eligible = sorted(set(classes) - duplicated, key=lambda x: sha(x.encode()))
    assert len(eligible) > 192
    fit, dev, test = (eligible[:64], eligible[64:128], eligible[128:])
    assert not set(fit) & set(dev) and (not set(test) & set(fit + dev))

    def ids(cs, view):
        return np.array([sorted(np.flatnonzero(classes == c), key=lambda i: files[i])[view] for c in cs])
    fit_source = np.r_[ids(fit, 1), ids(fit, 2)]
    fit_target = np.r_[ids(fit, 0), ids(fit, 0)]
    X = z['features'][fit_source].astype(np.float64)
    Y = z['features'][fit_target].astype(np.float64)
    tick = time.perf_counter()
    W = np.eye(X.shape[1]) + X.T @ np.linalg.solve(X @ X.T + np.eye(128), Y - X)
    fit_s = time.perf_counter() - tick
    E = z['features']
    predicted = norm((E.astype(np.float64) @ W).astype(np.float32))
    right = ids(test, 0)
    test_sources = np.r_[ids(test, 1), ids(test, 2)]
    density = np.partition(E[test_sources] @ E[right].T, len(test_sources) - 10, axis=0)[-10:].mean(axis=0)
    devrows = df.iloc[np.r_[ids(dev, 1), ids(dev, 2)]]
    audit = dict(dataset='RetailVisualRAG', original_classes=367, exact_pixel_duplicate_groups=[dict(hash=h, paths=[files[i] for i in ix]) for h, ix in groups.items() if len(ix) > 1], excluded_classes=sorted(duplicated), eligible_classes=len(eligible), fit_classes=fit, dev_classes=dev, test_classes=test, fit_pairs=128, identity_disjoint=True, selection='Class SHA256 order after exact-pixel duplicate exclusions; first filename target, other two separate source catalogs', fit_s=fit_s, ridge_lambda=1, protocol_sha256=sha(P.read_bytes()), source=json.loads((D / 'source.json').read_text()), feature_sha256=sha((O / 'retail_features.npz').read_bytes()))
    dump(O / 'retail_manifest.json', audit)
    blocks = np.array_split(np.arange(len(test)), 4)
    for view in [1, 2]:
        allleft = ids(test, view)
        for block, positions in enumerate(blocks):
            left = allleft[positions]
            gold = positions.copy()
            U, meta = build_queries(devrows, df.iloc[left])
            manifest = dict(sources=left.tolist(), targets=right.tolist(), gold=gold.tolist(), paths=files, queries=meta, protocol_sha256=sha(P.read_bytes()))
            dump(O / f'Retail_view{view}_block{block}_manifest.json', manifest)
            cosine = E[left] @ E[right].T
            candidates = {'cosine': top(cosine, 64), 'csls10': top(2 * cosine - density[None, :], 64)}
            scores = {'ridge128': predicted[left] @ E[right].T, 'visual_zero_label': cosine}
            for scorer, S in scores.items():
                for retriever, C in candidates.items():
                    for k in [1, 4, 16, 64]:
                        info = dict(dataset='RetailVisualRAG', seed=view, view=view, block=block, retriever=retriever, k=k, scorer=scorer, labels=128 if scorer == 'ridge128' else 0)
                        stem = f'Retail_view{view}_block{block}_{scorer}_{retriever}_k{k}'
                        evaluate_case(stem, S, C[:, :k], gold, U, meta, info)
    print('RETAIL COMPLETE', len(test), flush=True)
if __name__ == '__main__':
    main()
