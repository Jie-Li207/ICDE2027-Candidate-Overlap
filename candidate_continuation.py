import json, hashlib, gc, time
from pathlib import Path
import numpy as np
from confirmatory_data import workload, sha
from confirmatory_eval import R, O as OLD, decoders, metrics
O = R / 'candidate_continuation'
O.mkdir(exist_ok=True)

def main():
    for audit in sorted((R / 'frozen').glob('*_audit.json')):
        meta = json.loads(audit.read_text())
        ds = meta['dataset']
        stem = audit.name.removesuffix('_audit.json')
        man = json.loads((OLD / (ds.replace('/', '_') + '_identities.json')).read_text())
        part = np.array(man['test'][2048:2304])
        left = part[:, 0]
        goldtargets = part[:, 1]
        assert len(part) == 256
        assert not set(part.ravel()) & set(np.array(man['test'][:2048]).ravel())
        gallery = np.array(man['gallery'])
        others = np.array(sorted(set(gallery) - set(goldtargets)))
        others = others[np.random.default_rng(29001).permutation(len(others))]
        right = np.r_[goldtargets, others]
        gold = np.arange(256)
        sizes = sorted(set([n for n in [256, 512, 1024, 2048, 4096, 8192, len(gallery)] if n <= len(gallery)]))
        w = workload(ds, left, right, right)
        mf = O / (ds.replace('/', '_') + '_manifest.json')
        m = dict(dataset=ds, sources=left.tolist(), gold_targets=goldtargets.tolist(), nested_target_order=right.tolist(), sizes=sizes, disjoint=True, parent_seal_sha256=man['original_sealed_list_sha256'], protocol_sha256=sha((R / 'protocol_candidate_continuation.json').read_bytes()), query_keys={q: v[2] for q, v in w.items()})
        if mf.exists():
            assert json.loads(mf.read_text()) == json.loads(json.dumps(m))
        else:
            mf.write_text(json.dumps(m, indent=2))
        for mode in ['noimage'] if meta['noimage'] else ['initial', 'full']:
            dest = O / (stem + '_' + mode + '_results.json')
            if dest.exists():
                continue
            emb = [np.load(R / 'frozen' / (stem + f'_{mode}' + ('' if d == 0 else f'_depth{d}') + '_node_embeddings.npy'), mmap_mode='r') for d in range(3)]
            S = emb[0][left] @ emb[0][right].T
            base = S.copy()
            for e in emb[1:]:
                S += e[left] @ e[right].T
            S /= 3
            rows = []
            pred = {}
            for size in sizes:
                order = np.random.default_rng(29002 + size).permutation(size)
                gold_order = np.argsort(order)[:256]
                for depth, scores in [('depth0', base[:, order]), ('depth2', S[:, order])]:
                    for name, p, timing in decoders(scores, False):
                        pred[f'{depth}_{size}_{name}'] = p
                        for q, (U, V, keys) in w.items():
                            rows.append(dict(dataset=ds, seed=meta['seed'], mode=mode, score=depth, targets=size, distractor_ratio=(size - 256) / 256, decoder=name, query_set=q, **timing, **metrics(p, U, V[:, order], gold_order, q != 'cross_filter')))
            np.savez_compressed(dest.with_suffix('.npz'), **pred)
            dest.write_text(json.dumps(rows, indent=2))
            print('CANDIDATES ' + stem + ' ' + mode, flush=True)
            del emb, S, base
            gc.collect()
    rows = []
    for f in O.glob('*_results.json'):
        rows.extend(json.loads(f.read_text()))
    (O / 'all_results.json').write_text(json.dumps(rows))
    checks = []
    for ds in sorted(set((r['dataset'] for r in rows))):
        x = [r for r in rows if r['dataset'] == ds and r['mode'] == 'initial' and (r['score'] == 'depth2') and (r['query_set'] == 'coarse')]
        values = {(r['seed'], r['targets'], r['decoder']): r['micro_f1'] for r in x}
        seeds = sorted(set((r['seed'] for r in x)))
        gain = {n: float(np.mean([values[s, n, 'dense'] - values[s, n, 'row'] for s in seeds])) for n in [256, 1024]}
        checks.append(dict(dataset=ds, seeds=seeds, closed_gain=gain[256], fourfold_gallery_gain=gain[1024], relative_drop=(gain[256] - gain[1024]) / gain[256] if gain[256] else None, passes=bool(gain[256] > 0 and gain[1024] <= 0.5 * gain[256])))
    (O / 'primary_gate.json').write_text(json.dumps(dict(checks=checks, required_corpora=3, passed_corpora=sum((c['passes'] for c in checks)), overall_pass=sum((c['passes'] for c in checks)) >= 3), indent=2))
    print(json.dumps(checks, indent=2))
if __name__ == '__main__':
    main()
