import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from query_scope import R, O, P, sha, dump, top, evaluate_case, graph_queries

def jobs():
    for ds, seed, folder, stem in [('FBDB15K/norm', 0, 'cdmea_cpu', 'block0'), ('FBDB15K/norm', 1, 'cdmea_gpu', 'FBDB15K_norm_block1'), ('FBDB15K/norm', 2, 'cdmea_gpu', 'FBDB15K_norm_block2'), ('DBP15K/zh_en', 0, 'cdmea_gpu', 'DBP15K_zh_en_block0'), ('FBYG15K/norm', 0, 'cdmea_gpu', 'FBYG15K_norm_block0')]:
        for epoch in [50, 200]:
            yield (ds, seed, folder, stem, epoch)

def idle():
    while True:
        s = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True)
        util, mem = map(int, s.splitlines()[0].split(','))
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'], text=True)
        other = []
        for line in processes.splitlines():
            pid, name = line.split(',', 1)
            if int(pid) != os.getpid() and (not name.strip().lower().endswith('chatgpt.exe')):
                other.append(int(pid))
        if util < 15 and mem < 12000 and (not other):
            return
        print('WAIT GPU', util, mem, flush=True)
        time.sleep(15)

def export():
    import desalign_baseline as b
    import desalign_crosskg as cross
    from cdmea_cpu import models
    from cdmea_chunked import ChunkedRRGAT
    from cdmea_baseline import graph_arrays
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.25)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for ds, seed, folder, stem, epoch in jobs():
        name = ds.replace('/', '_') + f'_cdmea_seed{seed}_epoch{epoch}'
        if (O / (name + '_features.npz')).exists():
            continue
        idle()
        kgs, full, present, left, right, gold, U, G, meta = b.prepare(seed) if ds == 'FBDB15K/norm' else cross.prepare(ds, seed)
        a = SimpleNamespace(cuda=True, attr_dim=300, img_dim=300, hidden_units='300,300,300', heads='2,2', tau=0.1, dropout=0.0, gph_no_proxy=True, fusion_id=10, inner_view_num=4, with_weight=1, use_multi_loss=False, structure_encoder='RRGAT-gphimg', dist=2, csls_k=3)
        if kgs['att_features'].shape[1] < 1000:
            kgs['att_features'] = np.pad(kgs['att_features'], ((0, 0), (0, 1000 - kgs['att_features'].shape[1])))
        graph = graph_arrays(b.BASE / ds, kgs['ent_num'])
        graph = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in graph.items()}
        model = models.MultiModalEncoderDualNew(args=a, ent_num=kgs['ent_num'], img_feature_dim=full.shape[1], img_features=torch.nn.functional.normalize(torch.from_numpy(kgs['images_list']).cuda()), rel_features=torch.from_numpy(kgs['rel_features']).float().cuda(), att_features=torch.from_numpy(kgs['att_features']).float().cuda(), **graph).cuda()
        model.e_encoder.__class__ = ChunkedRRGAT
        model.r_encoder.__class__ = ChunkedRRGAT
        checkpoint = R / folder / (stem + f'_epoch{epoch}.pt')
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        tick = time.perf_counter()
        with torch.no_grad():
            _, img, _, _, joint = model.joint_emb_generate(only_joint=False)
            reference = -(model.compute_csls_dis(joint, left, right) - 0.2 * model.compute_csls_dis(img, left, right)).cpu().numpy() / 4
            cached = np.load(R / folder / (stem + f'_epoch{epoch}_initial_scores.npz'))['normalized']
            error = float(abs(reference - cached).max())
            assert error < 1e-05, error
            j = torch.nn.functional.normalize(joint).cpu().numpy()
            im = torch.nn.functional.normalize(img).cpu().numpy()

            def csls(x):
                a = x[left] @ x[right].T
                return 2 * a - np.partition(a, a.shape[1] - 3, axis=1)[:, -3:].mean(1)[:, None] - np.partition(a, a.shape[0] - 3, axis=0)[-3:].mean(0)[None, :]
            check = 0.5 * (csls(j) - 0.2 * csls(im)) - 0.2
            formula_error = float(abs(check - reference).max())
            assert formula_error < 1e-05, formula_error
        np.savez_compressed(O / (name + '_features.npz'), joint=j, image=im)
        dump(O / (name + '_export.json'), dict(dataset=ds, seed=seed, epoch=epoch, checkpoint_sha256=sha(checkpoint.read_bytes()), development_max_error=error, formula_max_error=formula_error, export_s=time.perf_counter() - tick, peak_gpu_bytes=torch.cuda.max_memory_allocated(), secondary_protocol_sha256=sha((R / 'protocol_cdmea_retrieval.json').read_bytes())))
        print('EXPORTED', name, error, formula_error, flush=True)
        del model, kgs, full, graph, img, joint, reference, j, im
        gc.collect()
        torch.cuda.empty_cache()

def density3(E, left, right, path):
    if path.exists():
        return np.load(path)
    best = np.full((3, len(right)), -np.inf, dtype=np.float32)
    targets = E[right]
    for start in range(0, len(left), 256):
        v = np.r_[best, E[left[start:start + 256]] @ targets.T]
        best = np.partition(v, len(v) - 3, axis=0)[-3:]
    out = best.mean(0)
    np.save(path, out)
    return out

def evaluate():
    from confirmatory_data import b
    protocol = json.loads(P.read_text())
    for ds, seed, folder, stem, epoch in jobs():
        tag = ds.replace('/', '_')
        name = tag + f'_cdmea_seed{seed}_epoch{epoch}'
        if (O / (name + '_block3_cdi_k64.json')).exists():
            continue
        z = np.load(O / (name + '_features.npz'))
        J = z['joint']
        I = z['image']
        man = json.loads((R / 'confirmatory' / (tag + '_identities.json')).read_text())
        start, stop = protocol['graph_cohorts'][ds]
        pairs = np.array(man['test'][start:stop])
        right = np.array(man['gallery'])
        fullleft = np.array([int(line.split(b'\t', 1)[0]) for line in (b.BASE / ds / 'ent_ids_1').read_bytes().splitlines() if line])
        lookup = {int(v): i for i, v in enumerate(right)}
        jd = density3(J, fullleft, right, O / (name + '_joint_density.npy'))
        imd = density3(I, fullleft, right, O / (name + '_image_density.npy'))
        dfolder = 'desalign_gpu' if ds == 'FBDB15K/norm' or (seed == 1 and ds == 'FBYG15K/norm') else 'desalign_cpu'
        prefix = dfolder + '__' + tag + f'_block{seed}_initial'
        emb = [np.load(R / 'frozen' / (prefix + ('' if d == 0 else f'_depth{d}') + '_node_embeddings.npy'), mmap_mode='r') for d in range(3)]
        for block in range(4):
            part = pairs[256 * block:256 * (block + 1)]
            left = part[:, 0]
            gold = np.array([lookup[int(v)] for v in part[:, 1]])
            U, meta = graph_queries(ds, left, np.array(man['dev'])[:, 0])
            tick = time.perf_counter()
            scores = 2 * (J[left] @ J[right].T) - jd - 0.2 * (2 * (I[left] @ I[right].T) - imd)
            C = top(scores, 64)
            search_s = time.perf_counter() - tick
            S = sum((e[left] @ e[right].T for e in emb)) / 3
            for k in [1, 4, 16, 64]:
                info = dict(dataset=ds, seed=seed, block=block, retriever=f'CDMEA{epoch}_CDI', k=k, scorer='DESAlign_depth2', candidate_labels=128, candidate_updates=epoch, retrieval_s=search_s, secondary_protocol_sha256=sha((R / 'protocol_cdmea_retrieval.json').read_bytes()))
                evaluate_case(name + f'_block{block}_cdi_k{k}', S, C[:, :k], gold, U, meta, info)
        del J, I, emb
        gc.collect()
    print('CDMEA EVALUATION COMPLETE', flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['export', 'evaluate'], required=True)
    args = parser.parse_args()
    export() if args.stage == 'export' else evaluate()
