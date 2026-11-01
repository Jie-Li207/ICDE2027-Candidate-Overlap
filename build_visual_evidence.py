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
O = R / 'visual'
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
    from build_study_figures import draw_motivation
    with plt.rc_context():
        draw_motivation(F)

def framework():
    from build_framework import draw
    draw(F)

def rewrites():
    rows = json.loads((R / 'query_diagnostics/mechanism_summary.json').read_text())
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
    w = np.load(R / 'retrieved_candidates' / f'{ds}_diagnostic256_workload.npz')
    i = 26
    assert int(w['left'][i]) == 9881 and int(w['right'][w['gold'][i]]) == 20381
    b = R / 'public_data/desalign/selected/desalign_datasets/DBP15K/fr_en'
    ids = {int(a): v for a, v in (x.split('\t') for x in (b / 'ent_ids_2').read_text(encoding='utf8').splitlines())}
    case = []
    for seed, folder in [(0, 'desalign_cpu'), (1, 'desalign_gpu')]:
        z = np.load(R / 'retrieved_candidates' / f'{ds}_seed{seed}_diagnostic256_results.npz')
        prefix = f'{folder}__{ds}_block{seed}_initial'
        ee = [np.load(R / 'frozen' / (prefix + ('' if d == 0 else f'_depth{d}') + '_node_embeddings.npy'), mmap_mode='r') for d in range(3)]
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
    (O / 'real_case.json').write_text(json.dumps({'dataset': 'DBP15K/fr_en', 'cohort': 'diagnostic256', 'source_index': i, 'source_id': 9881, 'test_manifest_index': 2074, 'selection': 'availability-selected public image case, no model tuning', 'images_origin': 'https://huggingface.co/datasets/HackCz/DBP15K_raw/blob/main/DBP_raw.zip', 'image_provenance': provenance, 'image_to_cached_feature_pixel_identity': 'unverified; dataset illustrations, not re-encoded for this figure', 'records': case}, indent=2))
    from build_study_figures import draw_case
    with plt.rc_context():
        draw_case(F)
ARCHIVE_SHA = '7c9fe6756291073967df15d6c5515c0d10457d0f1067dd7d9df51cdc68c55bb7'
if __name__ == '__main__':
    cases()
    motivation()
    rewrites()
    framework()
    print('Created four figures and checked the motivating example and recorded case.')
