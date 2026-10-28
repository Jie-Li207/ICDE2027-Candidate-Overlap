import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, hashlib, io, zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from scipy.optimize import linear_sum_assignment
from PIL import Image
from threadpoolctl import threadpool_limits
R = Path(__file__).resolve().parent
F = R / 'paper/figures'
O = R / 'visual_v43'
F.mkdir(parents=True, exist_ok=True)
O.mkdir(exist_ok=True)
threadpool_limits(2)
BLUE, ORANGE, GREEN, GRAY = ('#0072B2', '#D55E00', '#009E73', '#64748B')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})

def save(fig, name):
    fig.savefig(F / (name + '.pdf'), bbox_inches='tight', pad_inches=0.035)
    fig.savefig(F / (name + '.png'), dpi=180, bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)

def arrow(ax, a, b, color=GRAY, style='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=10, lw=1.2, color=color, linestyle=style))

def motivation():
    S = np.array([[0.8, 0.7, 0.1], [0.7, 0.95, 0.7], [0.1, 0.85, 0.75]])
    C = np.array([[0, 1], [0, 2], [1, 2]])
    A = np.full_like(S, -100)
    for i in range(3):
        A[i, C[i]] = S[i, C[i]]
    e = linear_sum_assignment(A, maximize=True)[1]
    p = linear_sum_assignment(S, maximize=True)[1]
    assert e.tolist() == [0, 2, 1] and p.tolist() == [0, 1, 2]
    fig = plt.figure(figsize=(7.05, 2.05))
    ax = fig.add_axes([0, 0.02, 0.33, 0.9])
    ax.set_xlim(-0.65, 2.0)
    ax.set_ylim(-0.6, 2.75)
    ax.axis('off')
    ax.text(-0.55, 2.6, '(a) One retrieved graph', weight='bold')
    for i in range(3):
        for j in C[i]:
            ax.plot([0, 1.3], [2 - i, 2 - j], color='#B5BDC7', lw=1.5, zorder=0)
    ax.plot([0, 1.3], [1, 1], color=ORANGE, lw=2, ls='--', zorder=1)
    for i in range(3):
        for x, name in [(0, 's'), (1.3, 't')]:
            ax.scatter([x], [2 - i], s=185, c='white', edgecolor=BLUE, zorder=3)
            ax.text(x, 2 - i, f'{name}{i + 1}', ha='center', va='center', fontsize=8, zorder=4)
    ax.text(-0.55, -0.5, 'True-edge recall: 2/3; pool recall: 3/3', fontsize=8)
    ax = fig.add_axes([0.34, 0.02, 0.31, 0.9])
    ax.axis('off')
    ax.text(0, 0.95, '(b) Two valid identity joins', weight='bold')
    for x, title, pred, color in [(0, 'Keep edges', e, BLUE), (0.55, 'Pool targets', p, ORANGE)]:
        ax.text(x, 0.76, title, color=color, weight='bold')
        for i, j in enumerate(pred):
            ax.text(x, 0.6 - i * 0.15, f's{i + 1}  →  t{j + 1}', fontsize=10)
    ax.text(0, 0.03, 'Identity accuracy: 1/3 → 3/3', fontsize=8)
    ax = fig.add_axes([0.68, 0.02, 0.32, 0.9])
    ax.axis('off')
    ax.text(0, 0.95, '(c) Query utility differs', weight='bold')
    ax.text(0, 0.76, 'Source group', fontsize=8)
    ax.text(0.53, 0.76, 'Edge / Pool F1', fontsize=8)
    ax.text(0, 0.55, '{s2, s3}')
    ax.text(0.53, 0.55, '100% / 100%')
    ax.text(0, 0.31, '{s2}')
    ax.text(0.53, 0.31, '0% / 100%')
    ax.text(0, 0.02, 'Broad groups hide within-group swaps.', fontsize=8)
    save(fig, 'motivation')
    (O / 'motivation_check.json').write_text(json.dumps({'synthetic_explanatory_example': True, 'scores': S.tolist(), 'edges': C.tolist(), 'gold': [0, 1, 2], 'edge_prediction': e.tolist(), 'pool_prediction': p.tolist(), 'new_pool_edge': [1, 1]}, indent=2))

def framework():
    fig, ax = plt.subplots(figsize=(7.05, 2.45))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.35)
    ax.axis('off')
    for x, title in [(0, '1  Evidence'), (2.65, '2  Candidates'), (5.4, '3  Identity relation'), (8, '4  Queries')]:
        ax.text(x, 3.15, title, weight='bold', fontsize=9)
    ax.text(0, 2.77, '128 labels; visual + graph', fontsize=7.8)
    for i, col in enumerate([BLUE, GREEN, GRAY]):
        for j in range(4):
            ax.add_patch(Rectangle((0.2 + j * 0.32, 2.1 - i * 0.25), 0.24, 0.16, color=col, alpha=0.3 + 0.15 * j))
    ax.text(0.1, 1.32, 'Freeze encoders / scores', fontsize=8)
    ax.text(0.1, 0.93, 'Restore source images', fontsize=8)
    arrow(ax, (1.8, 1.85), (2.55, 1.85))
    for r in range(3):
        for c in range(4):
            present = (r, c) in [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)]
            ax.add_patch(Rectangle((2.8 + c * 0.26, 2.1 - r * 0.24), 0.23, 0.2, facecolor=BLUE if present else '#F1F5F9', edgecolor='white'))
    ax.text(4.0, 1.97, 'E', fontsize=11, color=BLUE)
    ax.text(2.7, 1.25, 'Keep E or pool targets', fontsize=8)
    ax.text(2.7, 0.9, 'Oracle / full controls', fontsize=7.6)
    arrow(ax, (4.65, 1.85), (5.35, 1.85))
    ax.add_patch(Rectangle((5.5, 1.4), 1.7, 1.12, fill=False, edgecolor=GRAY, lw=0.9))
    ax.text(5.6, 2.28, 'Source', fontsize=8)
    ax.text(6.45, 2.28, 'Target', fontsize=8)
    for i, val in enumerate(['s1          t2', 's2          t1', 's3          t3']):
        ax.text(5.64, 2.02 - i * 0.22, val, fontsize=8)
    ax.text(5.4, 1.03, 'Row / edge / pool', fontsize=7.8)
    ax.text(5.4, 0.72, 'Keys and abstentions', fontsize=7.8)
    arrow(ax, (7.28, 1.85), (7.85, 1.85))
    ax.text(8, 2.57, 'Facts ⋈ M_p', fontsize=11, color=GREEN)
    ax.text(8, 2.1, 'DISTINCT target IDs', fontsize=8)
    for j, (label, color) in enumerate([('TP', GREEN), ('FP', ORANGE), ('FN', GRAY)]):
        ax.add_patch(Rectangle((8 + j * 0.58, 1.47), 0.48, 0.3, color=color, alpha=0.17))
        ax.text(8.24 + j * 0.58, 1.62, label, ha='center', va='center', fontsize=8)
    ax.text(8, 1.04, 'F1 by query stratum', fontsize=7.8)
    ax.text(8, 0.72, 'Time and memory', fontsize=7.8)
    ax.plot([0, 9.9], [0.48, 0.48], color='#BFC9D3', lw=0.8)
    ax.text(0.05, 0.1, 'Paired controls: same identities, supervision, scores and predicates; gold is used only for evaluation.', fontsize=8)
    save(fig, 'framework')

def rewrites():
    rows = json.loads((R / 'review_v40/mechanism_summary.json').read_text())
    fig, axs = plt.subplots(1, 4, figsize=(7.05, 2.35), sharey=True)
    for ax, ds, title in zip(axs, ['FBDB15K/norm', 'FBYG15K/norm', 'DBP15K/zh_en', 'DBP15K/fr_en'], ['FB–DB', 'FB–YG', 'ZH–EN', 'FR–EN']):
        rs = [x for x in rows if x['dataset'] == ds]
        base = np.zeros(3)
        for key, label, color in [('corrected', 'Corrected', GREEN), ('damaged', 'Damaged', ORANGE), ('wrong_to_wrong', 'Wrong → wrong', '#9DA9B5')]:
            vals = np.array([x[key] for x in rs]) * 100
            ax.bar(range(3), vals, bottom=base, color=color, width=0.65, label=label)
            base += vals
        collision = [100 * x['collision_rate'] for x in rs]
        ax.plot(range(3), collision, 'o--', color=BLUE, ms=3, lw=1, label='Row collisions')
        ax.set_xticks(range(3), ['n', '4n', 'Full'])
        ax.set_title(title)
        ax.set_ylim(0, 75)
        ax.grid(axis='y', alpha=0.18)
        ax.set_axisbelow(True)
        ax.set_xlabel('Gallery size')
    axs[0].set_ylabel('Sources (%)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc='upper center', bbox_to_anchor=(0.51, 1.05), frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93], pad=0.35, w_pad=0.6)
    save(fig, 'rewrites')

def cases():
    ds = 'DBP15K_fr_en'
    w = np.load(R / 'revision_v34' / f'{ds}_diagnostic256_workload.npz')
    i = 26
    assert int(w['left'][i]) == 9881 and int(w['right'][w['gold'][i]]) == 20381
    b = R / 'public_data/desalign/selected/desalign_datasets/DBP15K/fr_en'
    ids = {int(a): v for a, v in (x.split('\t') for x in (b / 'ent_ids_2').read_text(encoding='utf8').splitlines())}
    case = []
    for seed, folder in [(0, 'desalign_v11'), (1, 'desalign_v21_gpu')]:
        z = np.load(R / 'revision_v34' / f'{ds}_seed{seed}_diagnostic256_results.npz')
        prefix = f'{folder}__{ds}_block{seed}_initial'
        ee = [np.load(R / 'frozen_v27' / (prefix + ('' if d == 0 else f'_depth{d}') + '_node_embeddings.npy'), mmap_mode='r') for d in range(3)]
        s = sum((e[w['left'][i]] @ e[w['right']].T for e in ee)) / 3
        for method in ['oracle_closed_dense', 'full_dense', 'multimodal_depth0_k16_edge', 'multimodal_depth0_k16_union', 'visual_zero_label_k16_edge', 'visual_zero_label_k16_union']:
            pred = int(z[method + '_p'][i])
            C = z['visual_zero_label_k16_edges'][i]
            case.append({'seed': seed, 'method': method, 'target_index': pred, 'target_id': int(w['right'][pred]) if pred >= 0 else None, 'target_uri': ids[int(w['right'][pred])] if pred >= 0 else None, 'correct': bool(pred == w['gold'][i]), 'score': float(s[pred]) if pred >= 0 else None, 'gold_score': float(s[w['gold'][i]]), 'gold_visual_edge': bool(w['gold'][i] in C), 'prediction_visual_edge': bool(pred in C), 'gold_visual_pool': bool(w['gold'][i] in z['visual_zero_label_k16_edges'])})
    p = R / 'public_data/qualitative'
    if not (p / 'DBP_raw.zip').exists():
        import requests
        p.mkdir(parents=True, exist_ok=True)
        v = requests.get('https://huggingface.co/datasets/HackCz/DBP15K_raw/resolve/main/DBP_raw.zip', timeout=120)
        v.raise_for_status()
        (p / 'DBP_raw.zip').write_bytes(v.content)
    assert hashlib.sha256((p / 'DBP_raw.zip').read_bytes()).hexdigest() == ARCHIVE_SHA
    with zipfile.ZipFile(p / 'DBP_raw.zip') as z:
        inner = z.read('DBP15k_raw_all/DBP15k.zip')
    names = ['Frédéric_VIII_de_Danemark_fr.png', 'Frederick_VIII_of_Denmark_Frédéric_VIII_de_Danemark_en_fr.png', 'Haakon_VII_of_Norway_Frédéric_VIII_de_Danemark_en_fr.png']
    imgs = []
    provenance = []
    with zipfile.ZipFile(io.BytesIO(inner)) as z:
        for name in names:
            raw = z.read('DBP15k_raw/case/' + name)
            imgs.append(Image.open(io.BytesIO(raw)).convert('RGB'))
            provenance.append({'member': 'DBP15k_raw/case/' + name, 'sha256': hashlib.sha256(raw).hexdigest()})
    fig = plt.figure(figsize=(7.05, 2.45))
    titles = ['Source: Frédéric VIII', 'True target: Frederick VIII', 'Pool error: Haakon VII']
    for j, (im, title) in enumerate(zip(imgs, titles)):
        ax = fig.add_axes([0.005 + j * 0.19, 0.25, 0.175, 0.67])
        ax.imshow(im)
        ax.axis('off')
        ax.set_title(title, fontsize=7.8, pad=4)
    ax = fig.add_axes([0.59, 0.06, 0.41, 0.88])
    ax.axis('off')
    ax.text(0, 0.97, 'Same source; recorded decisions', fontsize=9, weight='bold')
    ax.text(0, 0.79, 'Decoder', fontsize=8)
    ax.text(0.65, 0.79, 'Seed 0 / 1', fontsize=8)
    for y, method, label in [(0.62, 'oracle_closed_dense', 'Oracle assignment'), (0.45, 'full_dense', 'Full assignment'), (0.28, 'multimodal_depth0_k16_union', 'MM pool, k=16'), (0.11, 'visual_zero_label_k16_union', 'Visual pool, k=16')]:
        vals = [next((x for x in case if x['method'] == method and x['seed'] == seed))['correct'] for seed in [0, 1]]
        ax.text(0, y, label, fontsize=8)
        for j, v in enumerate(vals):
            ax.text(0.68 + j * 0.18, y, 'Yes' if v else 'No', color=GREEN if v else ORANGE, fontsize=8, weight='bold')
    x = next((x for x in case if x['method'] == 'visual_zero_label_k16_union' and x['seed'] == 1))
    assert x['target_id'] == 13013 and (not x['prediction_visual_edge'])
    fig.text(0.01, 0.12, 'FR–EN, diagnostic source #26 (ID 9881); public dataset images.', fontsize=7.6)
    fig.text(0.01, 0.025, 'Seed 1 visual pooling chooses Haakon VII via an edge absent from its top-16 list.', fontsize=7.6)
    save(fig, 'real_case')
    (O / 'real_case.json').write_text(json.dumps({'dataset': 'DBP15K/fr_en', 'cohort': 'diagnostic256', 'source_index': i, 'source_id': 9881, 'test_manifest_index': 2074, 'selection': 'availability-selected public image case, no model tuning', 'images_origin': 'https://huggingface.co/datasets/HackCz/DBP15K_raw/blob/main/DBP_raw.zip', 'image_provenance': provenance, 'image_to_cached_feature_pixel_identity': 'unverified; dataset illustrations, not re-encoded for this figure', 'records': case}, indent=2))
ARCHIVE_SHA = '7c9fe6756291073967df15d6c5515c0d10457d0f1067dd7d9df51cdc68c55bb7'
if __name__ == '__main__':
    motivation()
    framework()
    rewrites()
    cases()
    print('Created four figures and checked the motivating example and recorded case.')
