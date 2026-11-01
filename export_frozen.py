import json, gc, argparse, time, hashlib
from pathlib import Path
import numpy as np, torch
import desalign_baseline as b
import desalign_crosskg as cross
from desalign_modality_control import NoImageDESAlign
R = Path(__file__).resolve().parent
O = R / 'frozen'
O.mkdir(exist_ok=True)

def export(dataset, seed, folder, noimage=False):
    stem = dataset.replace('/', '_') + f'_block{seed}' + ('_noimage' if noimage else '')
    checkpoint = R / folder / (stem + '_epoch50.pt')
    deststem = folder + '__' + stem
    if not checkpoint.exists():
        return False
    auditfile = O / (deststem + '_audit.json')
    modes = ['noimage'] if noimage else ['initial', 'full']
    expected = [O / (deststem + f'_{mode}' + ('' if d == 0 else f'_depth{d}') + '_node_embeddings.npy') for mode in modes for d in range(3)]
    if auditfile.exists() and all((p.exists() for p in expected)):
        assert json.loads(auditfile.read_text())['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'Checkpoint changed: use a fresh output directory'
        return True
    tick = time.perf_counter()
    kgs, full, present, left, right, gold, U, G, meta = b.prepare(seed) if dataset == 'FBDB15K/norm' else cross.prepare(dataset, seed)
    a = b.configs()
    a.data_choice, a.data_split = dataset.split('/')
    a.use_intermediate = int(a.data_choice == 'DBP15K')
    a.w_img = not noimage
    a.inner_view_num = 3 if noimage else 4
    model = (NoImageDESAlign if noimage else b.DESAlign)(kgs, a)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    original = model.img_features
    checks = []
    with torch.no_grad():
        for mode in ['noimage'] if noimage else ['initial', 'full']:
            if mode == 'full':
                replacement = original.clone()
                replacement[present] = torch.nn.functional.normalize(torch.from_numpy(full[present]))
                model.img_features = replacement
            emb, _ = model.joint_emb_generat()
            emb = torch.nn.functional.normalize(emb)
            ref = np.load(R / folder / (stem + f'_{mode}_scores.npz'))
            base = (emb[left] @ emb[right].T).numpy()
            scores = base.copy()
            np.save(O / (deststem + f'_{mode}_node_embeddings.npy'), emb.numpy())
            for depth in [1, 2]:
                emb = torch.nn.functional.normalize(torch.sparse.mm(model.adj, emb))
                scores += (emb[left] @ emb[right].T).numpy()
                np.save(O / (deststem + f'_{mode}_depth{depth}_node_embeddings.npy'), emb.numpy())
            errors = [float(np.max(abs(base - ref['depth0']))), float(np.max(abs(scores / 3 - ref['depth2'])))]
            assert max(errors) < 1e-05, errors
            checks.append(dict(mode=mode, development_score_max_errors=errors))
            model.img_features = original
    auditfile.write_text(json.dumps(dict(dataset=dataset, seed=seed, noimage=noimage, folder=folder, checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), elapsed_s=time.perf_counter() - tick, checks=checks), indent=2))
    print('EXPORTED ' + deststem + ' ' + str(checks), flush=True)
    del model, kgs
    gc.collect()
    return True

def jobs():
    for seed in range(3):
        yield ('FBDB15K/norm', seed, 'desalign_gpu')
    for seed in range(3):
        yield ('DBP15K/zh_en', seed, 'desalign_cpu')
    for ds in ['DBP15K/fr_en', 'FBYG15K/norm']:
        yield (ds, 0, 'desalign_cpu')
        yield (ds, 1, 'desalign_gpu')

def main():
    for ds, s, folder in jobs():
        for ni in [False, True]:
            export(ds, s, folder, ni)
if __name__ == '__main__':
    main()
