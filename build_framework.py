from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
R = Path(__file__).resolve().parent
INK = '#000000'
GRAY = '#78818C'
LINE = '#C7CCD3'
BLUE = '#476E9D'
ORANGE = '#B96C42'
GREEN = '#417D69'
NODE = ['#F4D685', '#ACD7DF', '#C6B8D9']
EDGE = {(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)}
POOL = {(i, j) for i in range(3) for j in range(3)}
ARCHIVE_SHA = '7c9fe6756291073967df15d6c5515c0d10457d0f1067dd7d9df51cdc68c55bb7'
IMAGES = [('Frédéric_VIII_de_Danemark_fr.png', '0751b9f81b9ad3a528614c84bdae1056050642572c21ac1db9768689ebb5715a'), ('Frederick_VIII_of_Denmark_Frédéric_VIII_de_Danemark_en_fr.png', '1636e5034afbf43e36c89ef27a71209ec8117538ba070a6a8ae2f4877d622fdf')]

def t(a, x, y, s, size=8.0, color=INK, bold=False, ha='left', **kwargs):
    return a.text(x, y, s, fontsize=size, color=INK if color == GRAY else color, fontweight='bold' if bold else 'normal', ha=ha, va='center', **kwargs)

def box(a, x, y, w, h, fill='white', edge=LINE, r=0.1, lw=0.65):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', facecolor=fill, edgecolor=edge, lw=lw))

def line(a, points, color=GRAY, lw=0.8, ls='-', z=2):
    a.plot([p[0] for p in points], [p[1] for p in points], color=color, lw=lw, ls=ls, zorder=z)

def arrow(a, x, y, xx, yy, color=GRAY, lw=0.9, ls='-'):
    a.add_patch(FancyArrowPatch((x, y), (xx, yy), arrowstyle='-|>', mutation_scale=8, lw=lw, color=color, linestyle=ls, zorder=4))

def graph(a, x, y, scale=1):
    coords = np.array([[0, 0.45], [0.37, 0.93], [0.82, 0.57], [0.65, 0.04], [0.08, -0.05]]) * scale + [x, y]
    for i, j in [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (0, 4)]:
        line(a, [coords[i], coords[j]], '#96A0AB', 0.65)
    for i, (xx, yy) in enumerate(coords):
        a.add_patch(Circle((xx, yy), 0.085 * scale, fc=NODE[i % 3], ec='#56606D', lw=0.65, zorder=4))

def bipartite(a, x, y, pooled=False):
    for i, j in sorted(POOL if pooled else EDGE):
        new = (i, j) not in EDGE
        line(a, [(x, y - i * 0.43), (x + 1.55, y - j * 0.43)], ORANGE if new else '#9DA5AE', 1.0 if new else 0.8, '--' if new else '-')
    for i in range(3):
        for xx, pre in [(x, 's'), (x + 1.55, 't')]:
            a.add_patch(Circle((xx, y - i * 0.43), 0.18, fc=NODE[i], ec='#515B67', lw=0.7, zorder=4))
            t(a, xx, y - i * 0.43, f'${pre}_{{{i + 1}}}$', 7.0, ha='center', zorder=5)

def matrix(a, x, y, pooled=False):
    for i in range(3):
        for j in range(3):
            present = (i, j) in EDGE
            new = pooled and (not present)
            fc = '#C5D7E9' if present else '#EDC7B1' if new else '#F0F1F3'
            a.add_patch(Rectangle((x + j * 0.32, y - i * 0.32), 0.28, 0.28, fc=fc, ec='white', lw=0.4))
            if new:
                t(a, x + j * 0.32 + 0.14, y - i * 0.32 + 0.14, '+', 8, ORANGE, True, 'center')
    t(a, x + 0.46, y + 0.48, 'targets', 6.9, GRAY, ha='center')

def relation(a, x, y, pooled=False):
    c = ORANGE if pooled else BLUE
    box(a, x, y, 1.66, 1.31, 'white', c, 0.025, 0.7)
    a.add_patch(Rectangle((x + 0.005, y + 0.98), 1.65, 0.32, fc='#FAEFE7' if pooled else '#EDF3F9', ec='none'))
    t(a, x + 0.41, y + 1.14, 'src', 7.2, c, True, 'center')
    t(a, x + 1.22, y + 1.14, 'tgt', 7.2, c, True, 'center')
    for i in range(3):
        yy = y + 0.8 - i * 0.28
        t(a, x + 0.41, yy, f'$s_{i + 1}$', 7.5, ha='center')
        t(a, x + 1.22, yy, f'$p(s_{i + 1})$', 7.5, ha='center')
        if i < 2:
            line(a, [(x + 0.12, yy - 0.15), (x + 1.54, yy - 0.15)], LINE, 0.4)

def load_images():
    p = R / 'public_data/qualitative/DBP_raw.zip'
    if not p.exists():
        raise FileNotFoundError('Run build_visual_evidence.py first to acquire the public case archive.')
    assert hashlib.sha256(p.read_bytes()).hexdigest() == ARCHIVE_SHA
    with zipfile.ZipFile(p) as z:
        inner = z.read('DBP15k_raw_all/DBP15k.zip')
    ims = []
    with zipfile.ZipFile(io.BytesIO(inner)) as z:
        for name, sha in IMAGES:
            data = z.read('DBP15k_raw/case/' + name)
            assert hashlib.sha256(data).hexdigest() == sha
            ims.append(Image.open(io.BytesIO(data)).convert('RGB'))
    return ims

def draw(output_dir=None):
    out = Path(output_dir) if output_dir else R / 'paper/figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(7.05, 3.03), facecolor='white')
    a = fig.add_axes([0.01, 0.025, 0.98, 0.95])
    a.set(xlim=(0, 16), ylim=(0, 6.85))
    a.axis('off')
    t(a, 0.05, 6.59, '(a) Evaluation path', 9.3, bold=True)
    for j, im in enumerate(load_images()):
        xx = 0.17 + j * 2.38
        ax = a.inset_axes([xx / 16, 5.02 / 6.85, 0.78 / 16, 1.12 / 6.85])
        ax.imshow(im)
        ax.axis('off')
        graph(a, xx + 1.03, 5.12, 0.86)
        t(a, xx + 0.91, 4.78, 'Source entities' if j == 0 else 'Target entities', 7.7, ha='center')
    box(a, 0.12, 4.03, 4.56, 0.49, '#F5F1DC', '#E6DFC5', 0.08, 0.65)
    t(a, 2.4, 4.275, 'Frozen multimodal scorers', 8.0, bold=True, ha='center')
    arrow(a, 1.02, 4.65, 1.02, 4.54)
    arrow(a, 3.4, 4.65, 3.4, 4.54)
    arrow(a, 2.4, 4.01, 2.4, 3.8)
    box(a, 0.12, 3.27, 4.56, 0.48, '#EAF1F7', '#D0DEEA', 0.08, 0.65)
    t(a, 2.4, 3.51, 'Retrieve Cᵢ  /  rerank with S', 8.0, BLUE, bold=True, ha='center')
    arrow(a, 2.4, 3.25, 2.4, 3.02)
    box(a, 0.12, 2.34, 4.56, 0.62, '#F5E2E7', '#D7ABB7', 0.1, 0.85)
    t(a, 2.31, 2.65, 'Candidate contract + decoder', 7.6, bold=True, ha='center')
    a.add_patch(Circle((4.64, 2.46), 0.22, fc='white', ec=GRAY, lw=1.1, zorder=6))
    a.add_patch(Circle((4.64, 2.46), 0.14, fc='none', ec=LINE, lw=0.7, zorder=7))
    line(a, [(4.8, 2.3), (5.09, 2.04)], GRAY, 2.0, z=7)
    arrow(a, 4.94, 2.57, 5.46, 2.57, GRAY, 1.3)
    arrow(a, 2.4, 2.32, 2.4, 2.11)
    box(a, 0.12, 1.58, 4.56, 0.48, '#F2EDF8', '#DAD0E7', 0.08, 0.65)
    t(a, 2.4, 1.82, 'Identity relation Mₚ', 8.0, bold=True, ha='center')
    arrow(a, 2.4, 1.56, 2.4, 1.35)
    box(a, 0.12, 0.72, 4.56, 0.58, '#E9F3E8', '#C8DCC6', 0.08, 0.65)
    t(a, 2.4, 1.01, 'Fact joins → distinct answer IDs', 7.8, GREEN, True, 'center')
    box(a, 5.53, 0.49, 10.36, 6.18, '#FCF4F5', '#DDC3CC', 0.22, 0.75)
    t(a, 5.84, 6.32, '(b) Candidate contract: same lists, different support', 9.0, bold=True)
    t(a, 5.84, 5.9, 'C₁ = {t1, t2}     C₂ = {t1, t3}     C₃ = {t2, t3}', 7.8, GRAY)
    for y, pooled, c, title in [(3.28, False, BLUE, 'Retain retrieved edges $E$'), (1.03, True, ORANGE, 'Pool candidate endpoints $R_E$')]:
        box(a, 5.82, y, 9.77, 2.04, 'white', '#DCDDE2', 0.13, 0.65)
        t(a, 6.07, y + 1.76, title, 8.2, c, True)
        bipartite(a, 6.31, y + 1.25, pooled)
        arrow(a, 8.19, y + 0.82, 8.7, y + 0.82, c)
        matrix(a, 9.01, y + 1.04, pooled)
        t(a, 9.47, y + 0.16, '$E$' if not pooled else '$L \\times R_E$', 7.7, c, True, 'center')
        arrow(a, 10.16, y + 0.82, 10.66, y + 0.82, c)
        t(a, 11.32, y + 1.03, 'Assignment', 7.5, c, True, 'center')
        t(a, 11.32, y + 0.63, 'original S', 7.1, GRAY, ha='center')
        arrow(a, 12.03, y + 0.82, 12.55, y + 0.82, c)
        relation(a, 12.77, y + 0.17, pooled)
        t(a, 14.92, y + 0.98, 'Mₚ', 8.5, c, True, 'center')
        t(a, 14.92, y + 0.56, '→ joins', 7.0, GRAY, ha='center')
    line(a, [(6.15, 0.75), (6.67, 0.75)], '#9DA5AE', 0.8)
    t(a, 6.8, 0.75, 'retrieved pair', 7.1, GRAY)
    line(a, [(10.0, 0.75), (10.52, 0.75)], ORANGE, 0.95, '--')
    t(a, 10.65, 0.75, 'newly admitted pair (+)', 7.1, ORANGE)
    fig.savefig(out / 'framework.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(out / 'framework.svg', metadata={'Date': None})
    fig.savefig(out / 'framework.png', dpi=300)
    plt.close(fig)
    audit = {'revision': 'candidate_support', 'illustrative_support': True, 'retrieved_edges': sorted(map(list, EDGE)), 'pooled_pairs': sorted(map(list, POOL)), 'newly_admitted_pairs': sorted(map(list, POOL - EDGE)), 'same_reranking_scores_within_candidate_comparison': True, 'retrieval_and_reranking_scores_not_assumed_equal': True, 'images_are_public_entity_illustrations': True, 'exact_cached_encoder_input_pixels': 'unverified', 'image_hashes': [{'file': n, 'sha256': s} for n, s in IMAGES], 'archive_sha256': ARCHIVE_SHA, 'gold_used_only_for_evaluation': True}
    (out / 'framework_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf8')
    return audit
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir')
    args = p.parse_args()
    draw(args.output_dir)
