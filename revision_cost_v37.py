import time, json, gc
import numpy as np
from scipy.optimize import linear_sum_assignment
from revision_v34 import R, faiss, partial
from export_frozen_v27 import jobs
O = R / 'revision_cost_v37'
O.mkdir(exist_ok=True)
for ds, seed, folder in jobs():
    if seed:
        continue
    path = O / (ds.replace('/', '_') + '.json')
    if path.exists():
        continue
    z = np.load(R / 'revision_v34' / (ds.replace('/', '_') + '_diagnostic256_workload.npz'))
    left = z['left']
    right = z['right']
    prefix = folder + '__' + ds.replace('/', '_') + '_block0_initial'
    E = [np.load(R / 'frozen_v27' / (prefix + ('' if d == 0 else f'_depth{d}') + '_node_embeddings.npy'), mmap_mode='r') for d in range(3)]
    times = []
    for rep in range(8):
        a = {}
        t = time.perf_counter()
        idx = faiss.IndexFlatIP(E[0].shape[1])
        idx.add(np.ascontiguousarray(E[0][right], dtype=np.float32))
        a['index_build'] = time.perf_counter() - t
        t = time.perf_counter()
        _, C = idx.search(np.ascontiguousarray(E[0][left], dtype=np.float32), 16)
        a['index_search'] = time.perf_counter() - t
        t = time.perf_counter()
        S = sum((e[left] @ e[right].T for e in E)) / 3
        a['full_rerank_matrix'] = time.perf_counter() - t
        t = time.perf_counter()
        row = C[np.arange(len(left)), S[np.arange(len(left))[:, None], C].argmax(1)]
        a['row'] = time.perf_counter() - t
        t = time.perf_counter()
        edge = partial(S, C)
        a['edge'] = time.perf_counter() - t
        t = time.perf_counter()
        union = np.unique(C)
        rr, cc = linear_sum_assignment(S[:, union], maximize=True)
        pool = union[cc]
        a['pool'] = time.perf_counter() - t
        if rep:
            times.append(a)
        del idx, S
        gc.collect()
    result = dict(dataset=ds, k=16, n=len(left), targets=len(right), retriever='multimodal_depth0', threads=2, warmups=1, repetitions=times, index_vector_bytes=int(len(right) * E[0].shape[1] * 4), summary={key: dict(median_s=float(np.median([a[key] for a in times])), q25_s=float(np.quantile([a[key] for a in times], 0.25)), q75_s=float(np.quantile([a[key] for a in times], 0.75))) for key in times[0]}, scope='Includes shared full-gallery rerank matrix; not candidate-only reranking or ANN. Index build is separated from per-batch search. SQL costs separately measured in systems_v28.')
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['summary']), flush=True)
    del E
    gc.collect()
