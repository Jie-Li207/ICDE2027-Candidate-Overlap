import json, hashlib, collections
from pathlib import Path
import numpy as np
import desalign_baseline as b
R = Path(__file__).resolve().parent
O = R / 'confirmatory'
O.mkdir(exist_ok=True)
DATASETS = ['FBDB15K/norm', 'FBYG15K/norm', 'DBP15K/zh_en', 'DBP15K/fr_en']

def sha(x):
    return hashlib.sha256(x).hexdigest()

def load_pairs(dataset):
    folder = b.BASE / dataset
    maps = []
    names = {}
    for side in [1, 2]:
        m = {name: int(i) for i, name in (s.split(b'\t', 1) for s in (folder / f'ent_ids_{side}').read_bytes().splitlines() if s)}
        maps.append(m)
        names.update({i: n for n, i in m.items()})
    if dataset == 'FBDB15K/norm':
        _, _, raw = b.entity_development.load()
        pairs = [(maps[0][a], maps[1][c]) for a, c in raw]
        sealed = [sha(a + c) for a, c in raw[1664:]]
        ref = json.loads((R / 'entity_development/manifest.json').read_text())['sealed_identity_hashes']
    else:
        a, s = dataset.split('/')
        pkl = a + '_id_img_feature_dict.pkl' if a.startswith('FB') else s + '_GA_id_img_feature_dict.pkl'
        with open(b.BASE / 'pkls' / pkl, 'rb') as f:
            images = b.NumpyOnly(f, encoding='latin1').load()
        hashes = {int(i): sha(np.asarray(x, dtype=np.float32).ravel().tobytes()) for i, x in images.items()}
        counts = collections.Counter(hashes.values())
        raw = np.loadtxt(folder / 'ill_ent_ids', dtype=np.int64)
        lc = collections.Counter(raw[:, 0])
        rc = collections.Counter(raw[:, 1])
        pairs = [tuple(map(int, p)) for p in raw if lc[p[0]] == rc[p[1]] == 1 and all((i in hashes and counts[hashes[i]] == 1 for i in p))]
        pairs.sort(key=lambda p: sha(names[p[0]] + b' ' + names[p[1]]))
        sealed = [sha(names[a] + b' ' + names[c]) for a, c in pairs[1664:]]
        ref = json.loads((R / 'desalign_cpu' / f"{dataset.replace('/', '_')}_block0_manifest.json").read_text())['sealed_pair_hashes']
    assert sealed == ref, 'Sealed identity list changed'
    pairs = np.array(pairs, np.int64)
    assert len(np.unique(pairs[:, 0])) == len(pairs) == len(np.unique(pairs[:, 1]))
    assert len(pairs) >= 3712
    train = pairs[:128]
    dev = pairs[128:1664]
    test = pairs[1664:]
    assert not set(train.ravel()) & set(dev.ravel())
    assert not set(np.r_[train.ravel(), dev.ravel()]) & set(test.ravel())
    gallery = np.array(sorted(set(maps[1].values()) - set(pairs[:1664, 1])), np.int64)
    assert set(test[:, 1]) <= set(gallery)
    return dict(dataset=dataset, eligible_count=len(pairs), sealed_hash_verified=True, original_sealed_count=len(sealed), original_sealed_list_sha256=sha(json.dumps(sealed).encode()), train=train.tolist(), dev=dev.tolist(), test=test.tolist(), gallery=gallery.tolist(), protocol_sha256=sha((R / 'protocol_confirmatory.json').read_bytes()))

def facts(dataset, side):
    T = np.loadtxt(b.BASE / dataset / f'triples_{side}', dtype=np.int64)
    exist = collections.defaultdict(set)
    value = collections.defaultdict(set)
    for h, r, t in T:
        exist[int(r)].add(int(h))
        value[int(r), int(t)].add(int(h))
    return (exist, value)

def predicates(pred, left, limit, mincount):
    ranked = []
    seen = set()
    selected = set(map(int, left))
    for key, members in pred.items():
        count = len(selected & members)
        if mincount <= count <= 0.5 * len(left):
            ranked.append((-count, key, members))
    out = []
    keys = []
    for _, key, members in sorted(ranked, key=lambda t: (t[0], t[1])):
        u = np.array([int(i) in members for i in left], bool)
        sig = u.tobytes()
        if sig in seen:
            continue
        seen.add(sig)
        out.append(u)
        keys.append(key)
        if len(out) == limit:
            break
    assert out
    return (np.array(out), keys)

def block_data(man, block, regime):
    part = np.array(man['test'][512 * block:512 * (block + 1)], np.int64)
    left = part[:, 0]
    if regime == 'closed':
        order = np.random.default_rng(27100 + block).permutation(len(part))
        right = part[order, 1]
        gold = np.argsort(order)
    else:
        right = np.array(man['gallery'])
        lookup = {i: j for j, i in enumerate(right)}
        gold = np.array([lookup[i] for i in part[:, 1]])
    return (left, right, gold)

def workload(dataset, left, right, target_selection_gallery):
    e, v = facts(dataset, 1)
    te, _ = facts(dataset, 2)
    coarse, ck = predicates(e, left, 8, 0.05 * len(left))
    value, vk = predicates(v, left, 64, 3)
    _, tk = predicates(te, target_selection_gallery, 8, 0.05 * len(target_selection_gallery))
    target = np.array([[int(i) in te[k] for i in right] for k in tk])
    q = []
    qkeys = []
    for i in range(min(8, len(coarse))):
        for shift in [0, 3]:
            j = (i + shift) % len(target)
            q.append((coarse[i], target[j]))
            qkeys.append([ck[i], tk[j]])
    return {'coarse': (coarse, np.ones((len(coarse), len(right)), bool), ck), 'value': (value, np.ones((len(value), len(right)), bool), vk), 'cross_filter': (np.array([x[0] for x in q]), np.array([x[1] for x in q]), qkeys)}

def main():
    for ds in DATASETS:
        m = load_pairs(ds)
        dest = O / (ds.replace('/', '_') + '_identities.json')
        text = json.dumps(m, indent=2)
        if dest.exists():
            assert json.loads(dest.read_text()) == m
        else:
            dest.write_text(text)
        for block in range(4):
            for regime in ['closed', 'expanded']:
                left, right, gold = block_data(m, block, regime)
                w = workload(ds, left, right, m['gallery'])
                arrays = dict(left=left, right=right, gold=gold)
                meta = {}
                for key, (U, V, keys) in w.items():
                    arrays[key + '_U'] = U
                    arrays[key + '_V'] = V
                    meta[key] = keys
                stem = ds.replace('/', '_') + f'_block{block}_{regime}'
                np.savez_compressed(O / (stem + '_workload.npz'), **arrays)
                (O / (stem + '_queries.json')).write_text(json.dumps(meta, indent=2))
        print('MANIFEST ' + ds + ' ' + str(m['eligible_count']), flush=True)
if __name__ == '__main__':
    main()
