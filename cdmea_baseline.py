import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, time, random, argparse, gc, hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
import desalign_baseline as data
from cdmea_cpu import models
from cdmea_memory import CheckpointRRGAT
import query_consensus as c
torch.set_num_threads(2)
R = Path(__file__).resolve().parent
O = R / 'cdmea_cpu'
O.mkdir(exist_ok=True)

def graph_arrays(folder, n):
    T = np.concatenate([np.loadtxt(folder / f'triples_{s}', dtype=np.int64) for s in [1, 2]])
    rel_size = int(T[:, 1].max()) + 2
    h = np.r_[T[:, 0], T[:, 2]]
    t = np.r_[T[:, 2], T[:, 0]]
    r = np.r_[T[:, 1] + 1, T[:, 1] + 1 + rel_size]
    order = np.lexsort((t, h))
    h = h[order]
    t = t[order]
    r = r[order]
    new = np.r_[True, (h[1:] != h[:-1]) | (t[1:] != t[:-1])]
    idx = np.cumsum(new) - 1
    counts = np.bincount(idx)
    adj = np.stack([h[new], t[new]])
    ri = np.stack([idx, r])
    rv = (1 / counts[idx]).astype(np.float32)
    ent_edges = np.concatenate([adj, np.stack([np.arange(n), np.arange(n)])], axis=1)
    ent_edges = np.unique(ent_edges.T, axis=0).T
    return dict(rel_size=2 * rel_size, triple_size=adj.shape[1], adj_matrix=torch.from_numpy(adj), r_index=torch.from_numpy(ri), r_val=torch.from_numpy(rv), rel_matrix=None, ent_matrix=torch.from_numpy(ent_edges))

def evaluate(model, full, present, left, right, gold, U, G, epoch, block):
    model.eval()
    original = model.img_features
    rows = []
    with torch.no_grad():
        for mode in ['initial', 'full']:
            if mode == 'full':
                replacement = original.clone()
                replacement[present] = torch.nn.functional.normalize(torch.from_numpy(full[present]))
                model.img_features = replacement
            tick = time.perf_counter()
            gph, img, rel, att, joint = model.joint_emb_generate(only_joint=False)
            dj = model.compute_csls_dis(joint, left, right)
            di = model.compute_csls_dis(img, left, right)
            official = -(dj - 0.2 * di).numpy()
            normalized = official / 4.0
            cos = (torch.nn.functional.normalize(joint[left]) @ torch.nn.functional.normalize(joint[right]).T).numpy()
            p = c.assignment(normalized)
            rows.append(dict(block=block, epoch=epoch, mode=mode, hits=float(np.mean(p == gold)), query_loss=c.summarize(c.answers(p, U), U, G)['query_loss'], inference_s=time.perf_counter() - tick))
            np.savez_compressed(O / f'block{block}_epoch{epoch}_{mode}_scores.npz', normalized=normalized, official=official, plain_cosine=cos, U=U, gold=gold)
            model.img_features = original
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--block', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    o = p.parse_args()
    kgs, full, present, left, right, gold, U, G, meta = data.prepare(o.block)
    a = SimpleNamespace(cuda=False, attr_dim=300, img_dim=300, hidden_units='300,300,300', heads='2,2', tau=0.1, dropout=0.0, gph_no_proxy=True, fusion_id=10, inner_view_num=4, with_weight=1, use_multi_loss=False, structure_encoder='RRGAT-gphimg', dist=2, csls_k=3)
    original_attr_dim = kgs['att_features'].shape[1]
    if original_attr_dim < 1000:
        kgs['att_features'] = np.pad(kgs['att_features'], ((0, 0), (0, 1000 - original_attr_dim)))
    meta['observed_attribute_dimensions'] = original_attr_dim
    meta['zero_padded_attribute_dimensions'] = kgs['att_features'].shape[1]
    random.seed(meta['seed'])
    np.random.seed(meta['seed'])
    torch.manual_seed(meta['seed'])
    graph = graph_arrays(data.BASE / 'FBDB15K/norm', kgs['ent_num'])
    model = models.MultiModalEncoderDualNew(args=a, ent_num=kgs['ent_num'], img_feature_dim=full.shape[1], img_features=torch.nn.functional.normalize(torch.from_numpy(kgs['images_list'])), rel_features=torch.from_numpy(kgs['rel_features']).float(), att_features=torch.from_numpy(kgs['att_features']).float(), **graph)
    model.e_encoder.__class__ = CheckpointRRGAT
    model.r_encoder.__class__ = CheckpointRRGAT
    del kgs['images_list']
    gc.collect()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0)
    meta.update(protocol_sha256=hashlib.sha256((R / 'protocol_cdmea.json').read_bytes()).hexdigest(), mode='smoke' if o.smoke else 'fixed200', graph_edges=graph['triple_size'])
    prefix = f'block{o.block}' + ('_smoke' if o.smoke else '')
    (O / f'{prefix}_manifest.json').write_text(json.dumps(meta, indent=2))
    logs = []
    results = []
    start = time.perf_counter()
    for epoch in range(1, 2 if o.smoke else 201):
        model.train()
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, detail = model(kgs['train_ill'])
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        logs.append(dict(epoch=epoch, loss=float(loss.detach()), epoch_s=time.perf_counter() - tick, elapsed_s=time.perf_counter() - start))
        (O / f'{prefix}_train.json').write_text(json.dumps(logs, indent=2))
        if epoch % 5 == 0 or o.smoke:
            print('TRAIN ' + json.dumps(logs[-1]), flush=True)
        if not o.smoke and epoch in [50, 200]:
            results.extend(evaluate(model, full, present, left, right, gold, U, G, epoch, o.block))
            (O / f'{prefix}_development.json').write_text(json.dumps(results, indent=2))
            torch.save(model.state_dict(), O / f'{prefix}_epoch{epoch}.pt')
            print('DEV ' + json.dumps(results[-2:]), flush=True)
    print('COMPLETE ' + prefix, flush=True)
if __name__ == '__main__':
    main()
