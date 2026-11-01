import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, time, gc, hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
import desalign_baseline as b
import desalign_crosskg as cross
from cdmea_cpu import models
from cdmea_memory import CheckpointRRGAT
from cdmea_baseline import graph_arrays
from confirmatory_eval import R, O, RESULT, evaluate
from research_queue import available

def jobs():
    for epoch in [50, 200]:
        yield ('FBDB15K/norm', 0, epoch, 'cdmea_cpu', 'block0')
    for ds, seed in [('FBDB15K/norm', 1), ('FBDB15K/norm', 2), ('DBP15K/zh_en', 0), ('DBP15K/fr_en', 0), ('FBYG15K/norm', 0)]:
        for epoch in [50, 200]:
            yield (ds, seed, epoch, 'cdmea_gpu', ds.replace('/', '_') + f'_block{seed}')

def run(ds, seed, epoch, folder, stem):
    checkpoint = R / folder / (stem + f'_epoch{epoch}.pt')
    done = O / (folder + '__' + stem + f'_epoch{epoch}_cdmea_complete.json')
    if not checkpoint.exists() or done.exists():
        return
    while available() < 10 * 1024 ** 3:
        time.sleep(15)
    kgs, full, present, left, right, gold, U, G, meta = b.prepare(seed) if ds == 'FBDB15K/norm' else cross.prepare(ds, seed)
    a = SimpleNamespace(cuda=False, attr_dim=300, img_dim=300, hidden_units='300,300,300', heads='2,2', tau=0.1, dropout=0.0, gph_no_proxy=True, fusion_id=10, inner_view_num=4, with_weight=1, use_multi_loss=False, structure_encoder='RRGAT-gphimg', dist=2, csls_k=3)
    if kgs['att_features'].shape[1] < 1000:
        kgs['att_features'] = np.pad(kgs['att_features'], ((0, 0), (0, 1000 - kgs['att_features'].shape[1])))
    graph = graph_arrays(b.BASE / ds, kgs['ent_num'])
    model = models.MultiModalEncoderDualNew(args=a, ent_num=kgs['ent_num'], img_feature_dim=full.shape[1], img_features=torch.nn.functional.normalize(torch.from_numpy(kgs['images_list'])), rel_features=torch.from_numpy(kgs['rel_features']).float(), att_features=torch.from_numpy(kgs['att_features']).float(), **graph)
    model.e_encoder.__class__ = CheckpointRRGAT
    model.r_encoder.__class__ = CheckpointRRGAT
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    original = model.img_features
    checks = []
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    del kgs['images_list']
    gc.collect()
    with torch.no_grad():
        for mode in ['initial', 'full']:
            if mode == 'full':
                repl = original.clone()
                repl[present] = torch.nn.functional.normalize(torch.from_numpy(full[present]))
                model.img_features = repl
            tick = time.perf_counter()
            gph, img, rel, att, joint = model.joint_emb_generate(only_joint=False)
            embed_s = time.perf_counter() - tick
            dev = -(model.compute_csls_dis(joint, left, right) - 0.2 * model.compute_csls_dis(img, left, right)).numpy() / 4
            ref = np.load(R / folder / (stem + f'_epoch{epoch}_{mode}_scores.npz'))['normalized']
            error = float(abs(dev - ref).max())
            assert error < 1e-05, error
            checks.append(dict(mode=mode, development_max_error=error, embedding_s=embed_s))
            jointnorm = torch.nn.functional.normalize(joint)
            for regime in ['closed', 'expanded']:
                for block in range(4):
                    z = np.load(O / (ds.replace('/', '_') + f'_block{block}_{regime}_workload.npz'))
                    li = z['left']
                    ri = z['right']
                    tick = time.perf_counter()
                    S = -(model.compute_csls_dis(joint, li, ri) - 0.2 * model.compute_csls_dis(img, li, ri)).numpy() / 4
                    cos = (jointnorm[li] @ jointnorm[ri].T).numpy()
                    score_s = time.perf_counter() - tick
                    for name, score in [('CDI_CSLS', S), ('joint_cosine', cos)]:
                        info = dict(dataset=ds, learner=f'CDMEA-{epoch}', training_device='gpu' if 'gpu' in folder else 'cpu', seed=seed, mode=mode, block=block, regime=regime, score=name, score_s=score_s, checkpoint_sha256=sha)
                        evaluate(score, z, info, RESULT / (folder + '__' + stem + f'_epoch{epoch}_{mode}_test{block}_{regime}_{name}.json'))
            model.img_features = original
    done.write_text(json.dumps(dict(dataset=ds, seed=seed, epoch=epoch, checks=checks), indent=2))
    del model, kgs, full
    gc.collect()

def main():
    for args in jobs():
        run(*args)
if __name__ == '__main__':
    main()
