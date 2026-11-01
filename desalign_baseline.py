import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys, json, time, pickle, hashlib, random, importlib.util, gc, argparse
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path[:0] = [str(R / 'desalign_cpu'), str(R / 'deps')]
import numpy as np, torch
torch.set_num_threads(2)
from threadpoolctl import threadpool_limits
threadpool_limits(2)
from model.DESAlign import DESAlign
from src.data import load_attr, load_relation
from src.utils import get_adjr, set_optim
import entity_externality as entity_development
BASE = R / 'public_data/desalign/selected/desalign_datasets'
OUT = R / 'desalign_development'
OUT.mkdir(exist_ok=True)

class NumpyOnly(pickle.Unpickler):

    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'), ('_codecs', 'encode')}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Disallowed global: {module}.{name}')
        return super().find_class(module, name)

def configs():
    spec = importlib.util.spec_from_file_location('upstream_cfg', R / 'thirdparty/DESAlign/config.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = m.cfg()
    old = sys.argv
    sys.argv = [old[0]]
    cfg.get_args()
    sys.argv = old
    a = cfg.cfg
    for k, v in dict(data_choice='FBDB15K', hidden_units='300,300,300', attr_dim=300, img_dim=300, name_dim=300, char_dim=300, hidden_size=300, intermediate_size=400, num_attention_heads=1, num_hidden_layers=1, use_intermediate=0, w_name=False, w_char=False, inner_view_num=4, lr=0.004, epoch=50, batch_size=128, stage=0, adapt='vae', awloss=1, scheduler='cos', warmup_ratio=0.15, device='cpu').items():
        setattr(a, k, v)
    return a

def prepare(block):
    data = BASE / 'FBDB15K/norm'
    maps = []
    for side in [1, 2]:
        maps.append({name: int(i) for i, name in (s.split(b'\t', 1) for s in (data / f'ent_ids_{side}').read_bytes().splitlines() if s)})
    ent = {**maps[0], **maps[1]}
    n = max(ent.values()) + 1
    assert len(set(ent.values())) == n
    triples = np.concatenate([np.loadtxt(data / f'triples_{s}', dtype=np.int64) for s in [1, 2]])
    rel = load_relation(n, triples, 1000)
    att = load_attr([str(data / 'training_attrs_1'), str(data / 'training_attrs_2')], n, {k.decode('utf8'): v for k, v in ent.items()}, 1000)
    adj = get_adjr(n, triples, norm=True).coalesce()
    with open(BASE / 'pkls/FBDB15K_id_img_feature_dict.pkl', 'rb') as f:
        images = NumpyOnly(f, encoding='latin1').load()
    dim = np.asarray(next(iter(images.values()))).size
    full = np.zeros((n, dim), np.float32)
    present = np.zeros(n, bool)
    for i, x in images.items():
        full[int(i)] = np.asarray(x).ravel()
        present[int(i)] = True
    rawidx, raw, pairs = entity_development.load()
    seed = 26101 + block
    train = np.array([(ent[a], ent[b]) for a, b in pairs[:128]], np.int64)
    rawtriples = [p[:3] for p in (s.split() for s in (entity_development.D / 'FB15K_EntityTriples.txt').read_bytes().splitlines()) if len(p) >= 3]
    nodes = sorted(set(rawidx['FB15K']) | {a for a, r, b in rawtriples} | {b for a, r, b in rawtriples})
    rng = np.random.default_rng(seed)
    rng.permutation(512)
    mask = rng.random(len(nodes)) >= 0.5
    observed = present.copy()
    for name, keep in zip(nodes, mask):
        if name in maps[0]:
            observed[maps[0][name]] = present[maps[0][name]] and keep
    audits = []
    reconciled = []
    for side, key in [(0, 'FB15K'), (1, 'DB15K')]:
        same = 0
        checked = 0
        cosines = []
        normalized_errors = []
        for name, k in rawidx[key].items():
            if name in maps[side] and present[maps[side][name]]:
                identical = np.array_equal(full[maps[side][name]], raw[key][k])
                checked += 1
                same += int(identical)
                aa = entity_development.norm(full[maps[side][name]][None])[0]
                bb = entity_development.norm(raw[key][k][None])[0]
                cosines.append(float(aa @ bb))
                normalized_errors.append(float(np.max(np.abs(aa - bb))))
                if not identical:
                    reconciled.append(dict(entity=name.decode('utf8', errors='replace'), processed_id=maps[side][name], original_image_key=k, processed_sha256=entity_development.digest(full[maps[side][name]].tobytes()), original_sha256=entity_development.digest(raw[key][k].tobytes())))
                    full[maps[side][name]] = raw[key][k]
        audits.append(dict(side=key, compared=checked, exact_vectors=same, cosine_min=float(np.min(cosines)), cosine_median=float(np.median(cosines)), normalized_max_error=max(normalized_errors)))
    (OUT / f'block{block}_feature_preflight.json').write_text(json.dumps(audits, indent=2))
    print('FEATURE_AUDIT ' + json.dumps(audits), flush=True)
    assert len(reconciled) == 4, 'Unexpected feature discrepancy beyond amendment feature_reconciliation'
    (OUT / f'block{block}_feature_reconciliation.json').write_text(json.dumps(reconciled, indent=2))
    mean = full[observed].mean(0)
    std = full[observed].std(0)
    initial = full.copy()
    missing = np.flatnonzero(~observed)
    fillrng = np.random.default_rng(seed + 1000)
    initial[missing] = fillrng.normal(mean, std, size=(len(missing), dim)).astype(np.float32)
    manifests = json.loads((R / 'entity_development/manifest.json').read_text())
    m = manifests['blocks'][block]
    left = np.array([ent[a.encode('latin1')] for a in m['sources']])
    right = np.array([ent[a.encode('latin1')] for a in m['targets']])
    gold = np.array(m['gold'])
    assert not set(train.ravel()) & set(np.r_[left, right])
    assert np.array_equal(np.flatnonzero(~observed[left]), m['hidden'])
    pred = {r: set() for r in m['queries']}
    for a, r, b in rawtriples:
        rk = r.decode('ascii')
        if rk in pred:
            pred[rk].add(a)
    U = np.array([[a.encode('latin1') in pred[q] for a in m['sources']] for q in m['queries']])
    G = entity_development.answers(gold, U)
    kgs = dict(ent_num=n, rel_num=int(triples[:, 1].max() + 1), images_list=initial, rel_features=rel, att_features=att, name_features=None, char_features=None, input_idx=torch.arange(n), ent_wo_img=missing, ent_w_img=np.flatnonzero(observed), adj=adj, train_ill=train)
    meta = dict(block=block, seed=seed, entities=n, triples=len(triples), dim=dim, train_pairs=len(train), development_pairs=len(left), hidden_dev=int((~observed[left]).sum()), feature_audit=audits, protocol_sha256=entity_development.digest((R / 'protocol_desalign_development.json').read_bytes()))
    return (kgs, full, present, left, right, gold, U, G, meta)

def evaluate(model, full, present, left, right, gold, U, G, epoch):
    model.eval()
    rows = []
    original = model.img_features
    with torch.no_grad():
        for mode in ['initial', 'full']:
            if mode == 'full':
                replacement = original.clone()
                replacement[present] = torch.nn.functional.normalize(torch.from_numpy(full[present]))
                model.img_features = replacement
            tick = time.perf_counter()
            emb, _ = model.joint_emb_generat()
            emb = torch.nn.functional.normalize(emb)
            scores = (emb[left] @ emb[right].T).numpy()
            base = scores.copy()
            t0 = time.perf_counter() - tick
            p = entity_development.assign(base)
            rows.append(dict(epoch=epoch, mode=mode, depth=0, hits=float(np.mean(p == gold)), row_hits=float(np.mean(base.argmax(1) == gold)), query_loss=entity_development.loss(p, U, G), inference_s=t0))
            for _ in range(2):
                emb = torch.nn.functional.normalize(torch.sparse.mm(model.adj, emb))
                scores += (emb[left] @ emb[right].T).numpy()
            scores /= 3
            p = entity_development.assign(scores)
            rows.append(dict(epoch=epoch, mode=mode, depth=2, hits=float(np.mean(p == gold)), row_hits=float(np.mean(scores.argmax(1) == gold)), query_loss=entity_development.loss(p, U, G), inference_s=time.perf_counter() - tick))
            if epoch == 50:
                np.savez_compressed(OUT / f'block{model.args.block}_{mode}_scores.npz', depth0=base, depth2=scores, U=U, gold=gold)
            model.img_features = original
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--prepare-only', action='store_true')
    options = parser.parse_args()
    block = options.block
    kgs, full, present, left, right, gold, U, G, meta = prepare(block)
    (OUT / f'block{block}_data_audit.json').write_text(json.dumps(meta, indent=2))
    print('DATA ' + json.dumps(meta), flush=True)
    if options.prepare_only:
        return
    a = configs()
    a.block = block
    random.seed(meta['seed'])
    np.random.seed(meta['seed'])
    torch.manual_seed(meta['seed'])
    model = DESAlign(kgs, a)
    (OUT / f'block{block}_config.json').write_text(json.dumps(vars(a), indent=2))
    del kgs['images_list']
    gc.collect()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    a.warmup_steps = int(0.15 * 50)
    a.total_steps = 50
    groups = [{'params': [p for n, p in model.named_parameters() if not any((s in n for s in ['weight_raw', 'bias', 'multi_loss_layer']))], 'lr': a.lr, 'weight_decay': a.weight_decay}, {'params': [p for n, p in model.named_parameters() if any((s in n for s in ['weight_raw', 'bias'])) and 'multi_loss_layer' not in n], 'lr': a.lr, 'weight_decay': 0.0}, {'params': [p for n, p in model.named_parameters() if 'multi_loss_layer' in n], 'lr': 5 * a.lr, 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(groups, eps=a.adam_epsilon)
    from transformers import get_cosine_schedule_with_warmup
    schedule = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=7, num_training_steps=50)
    logs = []
    results = []
    start = time.perf_counter()
    for epoch in range(1, 51):
        model.train()
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, detail = model(kgs['train_ill'])
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        optimizer.step()
        schedule.step()
        entry = dict(epoch=epoch, loss=float(loss.detach()), epoch_s=time.perf_counter() - tick, elapsed_s=time.perf_counter() - start)
        logs.append(entry)
        print('TRAIN ' + json.dumps(entry), flush=True)
        (OUT / f'block{block}_train.json').write_text(json.dumps(logs, indent=2))
        if epoch in [10, 25, 50]:
            results.extend(evaluate(model, full, present, left, right, gold, U, G, epoch))
            (OUT / f'block{block}_development.json').write_text(json.dumps(results, indent=2))
            print('DEV ' + json.dumps(results[-4:]), flush=True)
    torch.save(model.state_dict(), OUT / f'block{block}_epoch50.pt')
    print('COMPLETE block ' + str(block), flush=True)
if __name__ == '__main__':
    main()
