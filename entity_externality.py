import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys, json, hashlib, time, collections
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R / 'deps'))
import h5py, numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
threadpool_limits(2)
D = R / 'public_data/mmkb'
O = R / 'entity_development'
O.mkdir(exist_ok=True)

def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

def digest(x):
    return hashlib.sha256(x).hexdigest()

def load():
    idx = {}
    data = {}
    hashes = {}
    hc = collections.Counter()
    for side in ['FB15K', 'DB15K']:
        idx[side] = {a: b.decode('ascii') for a, b in (s.split(b'\t') for s in (D / f'{side}_ImageIndex.txt').read_bytes().splitlines() if s)}
        with h5py.File(D / f'{side}_visual.h5', 'r') as f:
            data[side] = {k: np.asarray(f[k]).ravel() for k in f}
        hashes[side] = {k: digest(v.tobytes()) for k, v in data[side].items()}
        hc.update(hashes[side].values())
    pairs = [(p[0], p[2]) for p in (s.split() for s in (D / 'sameAs.txt').read_bytes().splitlines()) if len(p) >= 4]
    fc = collections.Counter((a for a, b in pairs))
    dc = collections.Counter((b for a, b in pairs))
    pairs = [(a, b) for a, b in pairs if fc[a] == dc[b] == 1 and a in idx['FB15K'] and (b in idx['DB15K']) and (hc[hashes['FB15K'][idx['FB15K'][a]]] == 1) and (hc[hashes['DB15K'][idx['DB15K'][b]]] == 1)]
    pairs.sort(key=lambda p: digest(b' '.join(p)))
    return (idx, data, pairs)

def assign(s):
    return linear_sum_assignment(s, maximize=True)[1]

def answers(p, U):
    out = np.zeros((len(U), len(p)), dtype=bool)
    for q, u in enumerate(U):
        out[q, np.unique(p[u])] = True
    return out

def loss(p, U, G):
    return float(np.count_nonzero(answers(p, U) != G) / (2 * G.sum()))

def main():
    idx, data, pairs = load()
    triples = [tuple(s.split()[:3]) for s in (D / 'FB15K_EntityTriples.txt').read_bytes().splitlines() if len(s.split()) >= 3]
    nodes = sorted(set(idx['FB15K']) | {a for a, r, b in triples} | {b for a, r, b in triples})
    ni = {v: i for i, v in enumerate(nodes)}
    pred = collections.defaultdict(set)
    for a, r, b in triples:
        pred[r].add(a)
    present = np.array([a in idx['FB15K'] and np.linalg.norm(data['FB15K'][idx['FB15K'][a]]) > 0 for a in nodes])
    manifests = []
    for block, seed in enumerate([26101, 26102, 26103]):
        rng = np.random.default_rng(seed)
        pp = pairs[128 + 512 * block:128 + 512 * (block + 1)]
        source = [a for a, b in pp]
        target = [b for a, b in pp]
        order = rng.permutation(len(pp))
        target = [target[j] for j in order]
        gold = np.argsort(order)
        observed = present & (rng.random(len(nodes)) >= 0.5)
        si = np.array([ni[a] for a in source])
        queries = []
        for r, ent in sorted(pred.items(), key=lambda x: (-sum((a in x[1] for a in source)), x[0])):
            count = sum((a in ent for a in source))
            if 0.05 * len(pp) <= count <= 0.5 * len(pp):
                queries.append(r.decode('ascii'))
            if len(queries) == 8:
                break
        manifests.append(dict(block=block, seed=seed, sources=[a.decode('latin1') for a in source], targets=[b.decode('latin1') for b in target], gold=gold.tolist(), hidden=np.flatnonzero(~observed[si]).tolist(), queries=queries, observed_node_mask_sha256=digest(observed.tobytes())))
    (O / 'manifest.json').write_text(json.dumps(dict(eligible_pairs=len(pairs), blocks=manifests, sealed_identity_hashes=[digest(a + b) for a, b in pairs[1664:]], protocol_sha256=digest((R / 'protocol_entity_development.json').read_bytes())), indent=2))
if __name__ == '__main__':
    main()
