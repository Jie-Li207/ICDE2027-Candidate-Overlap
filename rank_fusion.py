import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, time, gc, shutil
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from retrieved_candidates import partial, per_query, summarize, digest, dump, KS
from export_frozen import jobs
R = Path(__file__).resolve().parent
O = R / 'rank_fusion'
O.mkdir(exist_ok=True)

def fusion(a, b):
    out = []
    for x, y in zip(a, b):
        scores = {}
        for ranked in [x, y]:
            for rank, j in enumerate(ranked, 1):
                scores[int(j)] = scores.get(int(j), 0.0) + 1 / (60 + rank)
        out.append(sorted(scores, key=lambda j: (-scores[j], j))[:128])
    return np.array(out)

def main():
    for ds, seed, folder in jobs():
        for cohort in ['diagnostic256', 'fresh64']:
            stem = ds.replace('/', '_') + f'_seed{seed}_{cohort}'
            dest = O / (stem + '_results.json')
            if dest.exists():
                continue
            wstem = ds.replace('/', '_') + '_' + cohort + '_workload'
            w = np.load(R / 'retrieved_candidates' / (wstem + '.npz'))
            wm = json.loads((R / 'retrieved_candidates' / (wstem + '.json')).read_text())
            left = w['left']
            right = w['right']
            gold = w['gold']
            n = len(left)
            for suffix in ['.npz', '.json']:
                if not (O / (wstem + suffix)).exists():
                    shutil.copy2(R / 'retrieved_candidates' / (wstem + suffix), O / (wstem + suffix))
            z = np.load(R / 'retrieved_candidates' / (stem + '_results.npz'))
            C128 = fusion(z['multimodal_depth0_k128_edges'], z['visual_zero_label_k128_edges'])
            prefix = folder + '__' + ds.replace('/', '_') + f'_block{seed}_initial'
            S = np.zeros((n, len(right)), np.float32)
            for depth in range(3):
                e = np.load(R / 'frozen' / (prefix + ('' if depth == 0 else f'_depth{depth}') + '_node_embeddings.npy'), mmap_mode='r')
                S += e[left] @ e[right].T
            S /= 3
            original = json.loads((R / 'retrieved_candidates' / (stem + '_results.json')).read_text())
            assert digest(S.tobytes()) == original['score_sha256']
            arrays = {}
            rows = []
            conditions = []
            for k in KS:
                C = C128[:, :k]
                union = np.unique(C)
                arrays[f'rrf60_k{k}_edges'] = C
                info = dict(retriever='rrf60', k=k, edge_recall=float(np.mean((C == gold[:, None]).any(1))), union_target_recall=float(np.isin(gold, union).mean()), union_size=len(union))
                p_row = C[np.arange(n), S[np.arange(n)[:, None], C].argmax(1)]
                p_edge = partial(S, C)
                p_pool = np.full(n, -1, int)
                rr, cc = linear_sum_assignment(S[:, union], maximize=True)
                p_pool[rr] = union[cc]
                for decoder, p in [('row', p_row), ('edge', p_edge), ('union', p_pool)]:
                    cid = f'rrf60_k{k}_{decoder}'
                    c = per_query(p, w['U'], w['V'], gold)
                    arrays[cid + '_p'] = p
                    arrays[cid + '_counts'] = c
                    extra = dict(**info, decoder=decoder, unsupported_edge_fraction=float(np.mean((p >= 0) & ~(C == p[:, None]).any(1))))
                    conditions.append(dict(id=cid, **extra))
                    rows.extend((dict(id=cid, dataset=ds, seed=seed, cohort=cohort, **extra, **x) for x in summarize(p, c, wm, gold)))
            np.savez_compressed(dest.with_suffix('.npz'), **arrays)
            dump(dest, dict(dataset=ds, seed=seed, cohort=cohort, score_sha256=digest(S.tobytes()), protocol_sha256=digest((R / 'protocol_fusion.json').read_bytes()), conditions=conditions, rows=rows))
            print('DONE', stem, flush=True)
            del S
            gc.collect()
if __name__ == '__main__':
    main()
