import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, hashlib, collections, time, random, argparse, gc
from pathlib import Path
import numpy as np, torch
import desalign_baseline as b
from desalign_modality_control import NoImageDESAlign
import query_consensus as consensus
R = Path(__file__).resolve().parent
O = R / 'desalign_cpu'
O.mkdir(exist_ok=True)

def sha(x):
    return hashlib.sha256(x).hexdigest()

def prepare(dataset, block):
    folder = b.BASE / dataset
    maps = []
    idnames = {}
    for side in [1, 2]:
        m = {name: int(i) for i, name in (s.split(b'\t', 1) for s in (folder / f'ent_ids_{side}').read_bytes().splitlines() if s)}
        maps.append(m)
        idnames.update({i: n for n, i in m.items()})
    n = max(idnames) + 1
    assert len(idnames) == n
    name = dataset.split('/')[0]
    suffix = dataset.split('/')[1]
    pkl = name + '_id_img_feature_dict.pkl' if name.startswith('FB') else suffix + '_GA_id_img_feature_dict.pkl'
    with open(b.BASE / 'pkls' / pkl, 'rb') as f:
        images = b.NumpyOnly(f, encoding='latin1').load()
    dim = np.asarray(next(iter(images.values()))).size
    full = np.zeros((n, dim), np.float32)
    present = np.zeros(n, bool)
    hashes = {}
    for i, x in images.items():
        i = int(i)
        full[i] = np.asarray(x).ravel()
        present[i] = True
        hashes[i] = sha(full[i].tobytes())
    counts = collections.Counter(hashes.values())
    raw = np.loadtxt(folder / 'ill_ent_ids', dtype=np.int64)
    lc = collections.Counter(raw[:, 0])
    rc = collections.Counter(raw[:, 1])
    pairs = [tuple(map(int, p)) for p in raw if lc[p[0]] == rc[p[1]] == 1 and all((present[i] and counts[hashes[i]] == 1 for i in p))]
    pairs.sort(key=lambda p: sha(idnames[p[0]] + b' ' + idnames[p[1]]))
    assert len(pairs) >= 1664
    train = np.array(pairs[:128], np.int64)
    part = np.array(pairs[128 + 512 * block:128 + 512 * (block + 1)], np.int64)
    seed = 26201 + block
    rng = np.random.default_rng(seed)
    order = rng.permutation(512)
    left = part[:, 0]
    right = part[order, 1]
    gold = np.argsort(order)
    assert not set(train.ravel()) & set(part.ravel())
    observed = present.copy()
    sources = np.array([i for name, i in sorted(maps[0].items())])
    observed[sources] &= rng.random(len(sources)) >= 0.5
    mean = full[observed].mean(0)
    std = full[observed].std(0)
    initial = full.copy()
    missing = np.flatnonzero(~observed)
    fill = np.random.default_rng(seed + 1000)
    initial[missing] = fill.normal(mean, std, (len(missing), dim)).astype(np.float32)
    triples = np.concatenate([np.loadtxt(folder / f'triples_{s}', dtype=np.int64) for s in [1, 2]])
    source_triples = np.loadtxt(folder / 'triples_1', dtype=np.int64)
    pred = collections.defaultdict(set)
    for a, r, c in source_triples:
        pred[int(r)].add(int(a))
    workload = []
    queryids = []
    for r, members in sorted(pred.items(), key=lambda x: (-sum((i in x[1] for i in left)), x[0])):
        u = np.array([i in members for i in left])
        if 0.05 * len(left) <= u.sum() <= 0.5 * len(left):
            workload.append(u)
            queryids.append(r)
        if len(workload) == 8:
            break
    assert len(workload) == 8
    U = np.array(workload)
    G = consensus.answers(gold, U)
    ent = {**maps[0], **maps[1]}
    rel = b.load_relation(n, triples, 1000)
    att = b.load_attr([str(folder / 'training_attrs_1'), str(folder / 'training_attrs_2')], n, {k.decode('utf8'): v for k, v in ent.items()}, 1000)
    adj = b.get_adjr(n, triples, norm=True).coalesce()
    kgs = dict(ent_num=n, rel_num=int(triples[:, 1].max() + 1), images_list=initial, rel_features=rel, att_features=att, name_features=None, char_features=None, input_idx=torch.arange(n), ent_wo_img=missing, ent_w_img=np.flatnonzero(observed), adj=adj, train_ill=train)
    meta = dict(dataset=dataset, block=block, seed=seed, n=n, image_dim=dim, images=int(present.sum()), gold_pairs=len(raw), eligible_pairs=len(pairs), excluded_pairs=len(raw) - len(pairs), duplicate_feature_groups=sum((c > 1 for c in counts.values())), train_pairs=train.tolist(), development_sources=left.tolist(), development_targets=right.tolist(), gold=gold.tolist(), query_ids=queryids, hidden_dev=int((~observed[left]).sum()), sealed_pair_hashes=[sha(idnames[a] + b' ' + idnames[c]) for a, c in pairs[1664:]], protocol_sha256=sha((R / 'protocol_crosskg_development.json').read_bytes()))
    return (kgs, full, present, left, right, gold, U, G, meta)

def evaluate(model, full, present, left, right, gold, U, G, stem, no_image):
    rows = []
    original = model.img_features
    model.eval()
    with torch.no_grad():
        for mode in ['noimage'] if no_image else ['initial', 'full']:
            if mode == 'full':
                replacement = original.clone()
                replacement[present] = torch.nn.functional.normalize(torch.from_numpy(full[present]))
                model.img_features = replacement
            t = time.perf_counter()
            emb, _ = model.joint_emb_generat()
            emb = torch.nn.functional.normalize(emb)
            base = (emb[left] @ emb[right].T).numpy()
            scores = base.copy()
            for _ in range(2):
                emb = torch.nn.functional.normalize(torch.sparse.mm(model.adj, emb))
                scores += (emb[left] @ emb[right].T).numpy()
            scores /= 3
            for depth, S in [(0, base), (2, scores)]:
                p = consensus.assignment(S)
                rows.append(dict(mode=mode, depth=depth, hits=float(np.mean(p == gold)), query_loss=consensus.summarize(consensus.answers(p, U), U, G)['query_loss']))
            np.savez_compressed(O / f'{stem}_{mode}_scores.npz', depth0=base, depth2=scores, U=U, gold=gold)
            rows[-1]['inference_s'] = time.perf_counter() - t
            model.img_features = original
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['FBYG15K/norm', 'DBP15K/zh_en', 'DBP15K/fr_en'], required=True)
    p.add_argument('--block', type=int, default=0)
    p.add_argument('--no-image', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    opt = p.parse_args()
    stem = opt.dataset.replace('/', '_') + f'_block{opt.block}' + ('_noimage' if opt.no_image else '')
    kgs, full, present, left, right, gold, U, G, meta = prepare(opt.dataset, opt.block)
    (O / f'{stem}_manifest.json').write_text(json.dumps(meta, indent=2))
    print('DATA ' + json.dumps({k: meta[k] for k in ['dataset', 'block', 'n', 'image_dim', 'eligible_pairs', 'excluded_pairs', 'hidden_dev']}), flush=True)
    if opt.prepare_only:
        return
    a = b.configs()
    a.data_choice = opt.dataset.split('/')[0]
    a.data_split = opt.dataset.split('/')[1]
    a.use_intermediate = int(a.data_choice == 'DBP15K')
    a.w_img = not opt.no_image
    a.inner_view_num = 3 if opt.no_image else 4
    random.seed(meta['seed'])
    np.random.seed(meta['seed'])
    torch.manual_seed(meta['seed'])
    model = (NoImageDESAlign if opt.no_image else b.DESAlign)(kgs, a)
    (O / f'{stem}_config.json').write_text(json.dumps(vars(a), indent=2))
    del kgs['images_list']
    gc.collect()
    groups = [{'params': [p for n, p in model.named_parameters() if not any((s in n for s in ['weight_raw', 'bias', 'multi_loss_layer']))], 'lr': a.lr, 'weight_decay': a.weight_decay}, {'params': [p for n, p in model.named_parameters() if any((s in n for s in ['weight_raw', 'bias'])) and 'multi_loss_layer' not in n], 'lr': a.lr, 'weight_decay': 0.0}, {'params': [p for n, p in model.named_parameters() if 'multi_loss_layer' in n], 'lr': 5 * a.lr, 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(groups, eps=a.adam_epsilon)
    from transformers import get_cosine_schedule_with_warmup
    schedule = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=7, num_training_steps=50)
    logs = []
    start = time.perf_counter()
    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(kgs['train_ill'])
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        optimizer.step()
        schedule.step()
        logs.append(dict(epoch=epoch, loss=float(loss.detach()), elapsed_s=time.perf_counter() - start))
        (O / f'{stem}_train.json').write_text(json.dumps(logs, indent=2))
        if epoch % 5 == 0:
            print('TRAIN ' + json.dumps(logs[-1]), flush=True)
    results = evaluate(model, full, present, left, right, gold, U, G, stem, opt.no_image)
    (O / f'{stem}_development.json').write_text(json.dumps(results, indent=2))
    torch.save(model.state_dict(), O / f'{stem}_epoch50.pt')
    print('COMPLETE ' + stem + ' ' + json.dumps(results), flush=True)
if __name__ == '__main__':
    main()
