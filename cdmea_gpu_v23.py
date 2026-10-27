import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, time, random, argparse, gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
import desalign_baseline_v9 as data
import desalign_crosskg_v11 as cross
from cdmea_cpu import models
from cdmea_memory import CheckpointRRGAT
from cdmea_chunked import ChunkedRRGAT
from cdmea_baseline_v14 import graph_arrays
from desalign_gpu_v21 import wait_idle
import query_consensus_v10 as c
R = Path(__file__).resolve().parent
O = R / 'cdmea_v23_gpu'
O.mkdir(exist_ok=True)

def evaluate(model, full, present, left, right, gold, U, G, epoch, stem):
    model.eval()
    original = model.img_features
    rows = []
    with torch.no_grad():
        for mode in ['initial', 'full']:
            if mode == 'full':
                replacement = original.clone()
                replacement[present] = torch.nn.functional.normalize(torch.from_numpy(full[present]).cuda())
                model.img_features = replacement
            torch.cuda.synchronize()
            tick = time.perf_counter()
            gph, img, rel, att, joint = model.joint_emb_generate(only_joint=False)
            dj = model.compute_csls_dis(joint, left, right)
            di = model.compute_csls_dis(img, left, right)
            official = -(dj - 0.2 * di).cpu().numpy()
            normalized = official / 4.0
            cos = (torch.nn.functional.normalize(joint[left]) @ torch.nn.functional.normalize(joint[right]).T).cpu().numpy()
            torch.cuda.synchronize()
            p = c.assignment(normalized)
            rows.append(dict(epoch=epoch, mode=mode, hits=float(np.mean(p == gold)), inference_s=time.perf_counter() - tick, **c.summarize(c.answers(p, U), U, G)))
            np.savez_compressed(O / f'{stem}_epoch{epoch}_{mode}_scores.npz', normalized=normalized, official=official, plain_cosine=cos, U=U, gold=gold)
            model.img_features = original
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--block', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    o = p.parse_args()
    stem = o.dataset.replace('/', '_') + f'_block{o.block}' + ('_smoke' if o.smoke else '')
    wait_idle()
    torch.cuda.set_per_process_memory_fraction(0.25)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kgs, full, present, left, right, gold, U, G, meta = data.prepare(o.block) if o.dataset == 'FBDB15K/norm' else cross.prepare(o.dataset, o.block)
    a = SimpleNamespace(cuda=True, attr_dim=300, img_dim=300, hidden_units='300,300,300', heads='2,2', tau=0.1, dropout=0.0, gph_no_proxy=True, fusion_id=10, inner_view_num=4, with_weight=1, use_multi_loss=False, structure_encoder='RRGAT-gphimg', dist=2, csls_k=3)
    original_attr_dim = kgs['att_features'].shape[1]
    if original_attr_dim < 1000:
        kgs['att_features'] = np.pad(kgs['att_features'], ((0, 0), (0, 1000 - original_attr_dim)))
    meta.update(observed_attribute_dimensions=original_attr_dim, device='cuda', gpu_allocator_fraction=0.25, tf32=False)
    random.seed(meta['seed'])
    np.random.seed(meta['seed'])
    torch.manual_seed(meta['seed'])
    torch.cuda.manual_seed_all(meta['seed'])
    graph = graph_arrays(data.BASE / o.dataset, kgs['ent_num'])
    graph = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in graph.items()}
    model = models.MultiModalEncoderDualNew(args=a, ent_num=kgs['ent_num'], img_feature_dim=full.shape[1], img_features=torch.nn.functional.normalize(torch.from_numpy(kgs['images_list']).cuda()), rel_features=torch.from_numpy(kgs['rel_features']).float().cuda(), att_features=torch.from_numpy(kgs['att_features']).float().cuda(), **graph).cuda()
    model.e_encoder.__class__ = CheckpointRRGAT
    model.r_encoder.__class__ = CheckpointRRGAT
    model.e_encoder.__class__ = ChunkedRRGAT
    model.r_encoder.__class__ = ChunkedRRGAT
    del kgs['images_list']
    gc.collect()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0)
    (O / f'{stem}_manifest.json').write_text(json.dumps(meta, indent=2))
    logs = []
    results = []
    start = time.perf_counter()
    for epoch in range(1, 2 if o.smoke else 201):
        wait_idle()
        model.train()
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(kgs['train_ill'])
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        logs.append(dict(epoch=epoch, loss=float(loss.detach()), compute_s=time.perf_counter() - tick, elapsed_s=time.perf_counter() - start, peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        (O / f'{stem}_train.json').write_text(json.dumps(logs, indent=2))
        time.sleep(1)
        if epoch % 5 == 0 or o.smoke:
            print('TRAIN ' + json.dumps(logs[-1]), flush=True)
        if not o.smoke and epoch in [50, 200]:
            results.extend(evaluate(model, full, present, left, right, gold, U, G, epoch, stem))
            (O / f'{stem}_development.json').write_text(json.dumps(results, indent=2))
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, O / f'{stem}_epoch{epoch}.pt')
            print('DEV ' + json.dumps(results[-2:]), flush=True)
    print('COMPLETE ' + stem, flush=True)
if __name__ == '__main__':
    main()
