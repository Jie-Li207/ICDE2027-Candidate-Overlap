import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, argparse, time, random, gc, subprocess
from pathlib import Path
import numpy as np, torch
import desalign_baseline_v9 as b
import desalign_crosskg_v11 as cross
from desalign_gpu.model.DESAlign import DESAlign
import query_consensus_v10 as c
R = Path(__file__).resolve().parent
O = R / 'desalign_v21_gpu'
O.mkdir(exist_ok=True)

class NoImage(DESAlign):

    def inner_view_loss(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill, vir_emb=None, weight_norm=None):
        if weight_norm is None or self.args.w_img:
            return super().inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill, vir_emb, weight_norm)
        assert img_emb is None and vir_emb is None and (name_emb is None) and (char_emb is None) and (weight_norm.shape[1] == 3)
        w = weight_norm * 3
        return self.multi_loss_layer([self.criterion_cl(gph_emb, train_ill, weight_norm=w[:, 2]), self.criterion_cl(rel_emb, train_ill, weight_norm=w[:, 1]), self.criterion_cl(att_emb, train_ill, weight_norm=w[:, 0]), 0, 0, 0])

def wait_idle():
    while True:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        free, util = map(int, r.stdout.strip().split(','))
        if free >= 8192 and util <= 40:
            return
        time.sleep(5)

def evaluate(model, full, present, left, right, gold, U, G, stem, noimage):
    rows = []
    original = model.img_features
    model.eval()
    with torch.no_grad():
        for mode in ['noimage'] if noimage else ['initial', 'full']:
            if mode == 'full':
                replacement = original.clone()
                replacement[present] = torch.nn.functional.normalize(torch.from_numpy(full[present]).cuda())
                model.img_features = replacement
            torch.cuda.synchronize()
            start = time.perf_counter()
            emb, _ = model.joint_emb_generat()
            emb = torch.nn.functional.normalize(emb)
            np.save(O / f'{stem}_{mode}_node_embeddings.npy', emb.cpu().numpy())
            base = (emb[left] @ emb[right].T).cpu().numpy()
            scores = base.copy()
            for _ in range(2):
                emb = torch.nn.functional.normalize(torch.sparse.mm(model.adj, emb))
                scores += (emb[left] @ emb[right].T).cpu().numpy()
            scores /= 3
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            for depth, S in [(0, base), (2, scores)]:
                p = c.assignment(S)
                rows.append(dict(mode=mode, depth=depth, hits=float(np.mean(p == gold)), inference_s=elapsed, **c.summarize(c.answers(p, U), U, G)))
            np.savez_compressed(O / f'{stem}_{mode}_scores.npz', depth0=base, depth2=scores, U=U, gold=gold)
            model.img_features = original
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--block', type=int, default=0)
    p.add_argument('--no-image', action='store_true')
    p.add_argument('--smoke', action='store_true')
    opt = p.parse_args()
    stem = opt.dataset.replace('/', '_') + f'_block{opt.block}' + ('_noimage' if opt.no_image else '') + ('_smoke' if opt.smoke else '')
    wait_idle()
    torch.cuda.set_per_process_memory_fraction(0.25)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kgs, full, present, left, right, gold, U, G, meta = b.prepare(opt.block) if opt.dataset == 'FBDB15K/norm' else cross.prepare(opt.dataset, opt.block)
    a = b.configs()
    a.data_choice = opt.dataset.split('/')[0]
    a.data_split = opt.dataset.split('/')[1]
    a.use_intermediate = int(a.data_choice == 'DBP15K')
    a.w_img = not opt.no_image
    a.inner_view_num = 3 if opt.no_image else 4
    a.device = 'cuda'
    random.seed(meta['seed'])
    np.random.seed(meta['seed'])
    torch.manual_seed(meta['seed'])
    torch.cuda.manual_seed_all(meta['seed'])
    model = (NoImage if opt.no_image else DESAlign)(kgs, a).cuda()
    del kgs['images_list']
    gc.collect()
    (O / f'{stem}_manifest.json').write_text(json.dumps(meta, indent=2))
    (O / f'{stem}_config.json').write_text(json.dumps(vars(a), indent=2))
    groups = [{'params': [p for n, p in model.named_parameters() if not any((s in n for s in ['weight_raw', 'bias', 'multi_loss_layer']))], 'lr': a.lr, 'weight_decay': a.weight_decay}, {'params': [p for n, p in model.named_parameters() if any((s in n for s in ['weight_raw', 'bias'])) and 'multi_loss_layer' not in n], 'lr': a.lr, 'weight_decay': 0.0}, {'params': [p for n, p in model.named_parameters() if 'multi_loss_layer' in n], 'lr': 5 * a.lr, 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(groups, eps=a.adam_epsilon)
    from transformers import get_cosine_schedule_with_warmup
    schedule = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=7, num_training_steps=50)
    logs = []
    start = time.perf_counter()
    for epoch in range(1, 2 if opt.smoke else 51):
        wait_idle()
        t = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(kgs['train_ill'])
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        optimizer.step()
        schedule.step()
        torch.cuda.synchronize()
        logs.append(dict(epoch=epoch, loss=float(loss.detach()), compute_s=time.perf_counter() - t, elapsed_s=time.perf_counter() - start, peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        (O / f'{stem}_train.json').write_text(json.dumps(logs, indent=2))
        time.sleep(1)
        if epoch % 5 == 0 or opt.smoke:
            print('TRAIN ' + json.dumps(logs[-1]), flush=True)
    if opt.smoke:
        return
    rows = evaluate(model, full, present, left, right, gold, U, G, stem, opt.no_image)
    (O / f'{stem}_development.json').write_text(json.dumps(rows, indent=2))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, O / f'{stem}_epoch50.pt')
    print('COMPLETE ' + stem + ' ' + json.dumps(rows), flush=True)
if __name__ == '__main__':
    main()
