from pathlib import Path
import argparse, hashlib, io, itertools, json, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Polygon
from PIL import Image
from scipy.optimize import linear_sum_assignment
R = Path(__file__).resolve().parent
BLUE = '#296E9C'
ORANGE = '#B85E31'
GREEN = '#227549'
RED = '#B31B1B'
NODE = ['#EBCF7A', '#A9D5DB', '#C9B6DD']
ARCHIVE_SHA = '7c9fe6756291073967df15d6c5515c0d10457d0f1067dd7d9df51cdc68c55bb7'
MEMBERS = [('Frédéric_VIII_de_Danemark_fr.png', '0751b9f81b9ad3a528614c84bdae1056050642572c21ac1db9768689ebb5715a'), ('Frederick_VIII_of_Denmark_Frédéric_VIII_de_Danemark_en_fr.png', '1636e5034afbf43e36c89ef27a71209ec8117538ba070a6a8ae2f4877d622fdf'), ('Haakon_VII_of_Norway_Frédéric_VIII_de_Danemark_en_fr.png', 'f19844ecc1384900e536b25b32009c05587f2f4cc276eee571d15be42ddc2657')]
METHODS = [('oracle_closed_dense', 'Oracle'), ('full_dense', 'Full gallery'), ('multimodal_depth0_k16_edge', 'MM edges'), ('multimodal_depth0_k16_union', 'MM pool'), ('visual_zero_label_k16_edge', 'Visual edges'), ('visual_zero_label_k16_union', 'Visual pool')]
try:
    findfont('Times New Roman', fallback_to_default=False)
    FONT = 'Times New Roman'
except ValueError:
    FONT = 'STIXGeneral'

def txt(a, x, y, s, size=8.5, bold=False, ha='left', color='black', **kw):
    return a.text(x, y, s, fontsize=size, fontweight='bold' if bold else 'normal', ha=ha, va='center', color=color, **kw)

def line(a, x, y, xx, yy, color='#AEB6BF', lw=0.8, ls='-', z=1):
    a.plot([x, xx], [y, yy], color=color, lw=lw, ls=ls, zorder=z)

def arrow(a, x, y, xx, yy, color='black', lw=0.9, ls='-'):
    direction = np.array([xx - x, yy - y], dtype=float)
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    tip = np.array([xx, yy])
    base = tip - 0.18 * direction
    a.plot([x, base[0] + 0.02 * direction[0]], [y, base[1] + 0.02 * direction[1]], color=color, lw=lw, ls=ls, solid_capstyle='round', dash_capstyle='round', zorder=2)
    if ls != '-':
        neck = base - 0.06 * direction
        join = base + 0.02 * direction
        a.plot([neck[0], join[0]], [neck[1], join[1]], color=color, lw=lw, solid_capstyle='round', zorder=2)
    a.add_patch(Polygon([tip, base + 0.065 * normal, base - 0.065 * normal], closed=True, facecolor=color, edgecolor='none', zorder=3))

def node(a, x, y, label, color, r=0.235, size=8.3):
    c = Circle((x, y), r, fc=color, ec='#252525', lw=0.7, zorder=4)
    a.add_patch(c)
    txt(a, x, y, label, size, ha='center', zorder=5)

def photo(a, im, x, y, w, h, H):
    ax = a.inset_axes([x / 16, y / H, w / 16, h / H])
    ax.imshow(im)
    ax.axis('off')

def images():
    p = R / 'public_data/qualitative/DBP_raw.zip'
    if not p.exists():
        raise FileNotFoundError('Run build_visual_evidence.py first to acquire the public image archive.')
    assert hashlib.sha256(p.read_bytes()).hexdigest() == ARCHIVE_SHA
    with zipfile.ZipFile(p) as z:
        data = z.read('DBP15k_raw_all/DBP15k.zip')
    ims = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for n, h in MEMBERS:
            b = z.read('DBP15k_raw/case/' + n)
            assert hashlib.sha256(b).hexdigest() == h
            ims.append(Image.open(io.BytesIO(b)).convert('RGB'))
    return ims

def canvas(H):
    plt.rcParams.update({'font.family': FONT, 'font.size': 8.5, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'text.color': 'black'})
    f = plt.figure(figsize=(7.05, 7.05 * H / 16), facecolor='white')
    a = f.add_axes([0.012, 0.02, 0.976, 0.96])
    a.set(xlim=(0, 16), ylim=(0, H))
    a.axis('off')
    return (f, a)

def save(f, name, output_dir):
    out = Path(output_dir) if output_dir else R / 'paper/figures'
    out.mkdir(parents=True, exist_ok=True)
    f.savefig(out / (name + '.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    f.savefig(out / (name + '.svg'), metadata={'Date': None})
    f.savefig(out / (name + '.png'), dpi=300)
    plt.close(f)
    return out

def draw_motivation(output_dir=None):
    S = np.array([[0.8, 0.7, 0.1], [0.7, 0.95, 0.7], [0.1, 0.85, 0.75]])
    C = np.array([[0, 1], [0, 2], [1, 2]])
    E = {(i, int(j)) for i in range(3) for j in C[i]}
    A = np.full_like(S, -100.0)
    for i in range(3):
        A[i, C[i]] = S[i, C[i]]
    edge = linear_sum_assignment(A, maximize=True)[1]
    pool = linear_sum_assignment(S, maximize=True)[1]
    assert edge.tolist() == [0, 2, 1] and pool.tolist() == [0, 1, 2]
    for pred, allowed in [(edge, E), (pool, {(i, j) for i in range(3) for j in range(3)})]:
        candidates = [p for p in itertools.permutations(range(3)) if all(((i, j) in allowed for i, j in enumerate(p)))]
        assert tuple(pred) == max(candidates, key=lambda p: sum((S[i, j] for i, j in enumerate(p))))
    H = 6.05
    f, a = canvas(H)
    ims = images()
    txt(a, 0.08, 5.82, '(a) Visual identity and source predicates', 9.5, True)
    a.add_patch(FancyBboxPatch((0.1, 3.11), 7.45, 2.35, boxstyle='round,pad=0,rounding_size=.16', fc='#FFF3D0', ec='none'))
    a.add_patch(FancyBboxPatch((8.35, 3.11), 7.55, 2.35, boxstyle='round,pad=0,rounding_size=.16', fc='#FBE7D9', ec='none'))
    txt(a, 3.62, 3.43, 'Source entity', 9.0, True, ha='center')
    txt(a, 13.15, 5.22, 'Target entities', 9.0, True, ha='center')
    node(a, 3.62, 4.12, '$s_2$', NODE[1], 0.25, 9)
    node(a, 1.1, 4.67, '$U_b$', '#CAE2C3', 0.25, 8.8)
    node(a, 1.1, 3.67, '$U_n$', '#EADAA2', 0.25, 8.8)
    arrow(a, 1.37, 4.62, 3.33, 4.18, GREEN)
    arrow(a, 1.37, 3.73, 3.33, 4.04, GREEN)
    txt(a, 2.13, 4.7, 'contains', 7.5, ha='center', color=GREEN)
    txt(a, 2.13, 3.53, 'contains', 7.5, ha='center', color=GREEN)
    txt(a, 0.42, 5.07, '$U_b=\\{s_2,s_3\\}$', 8.5)
    txt(a, 0.48, 3.26, '$U_n=\\{s_2\\}$', 8.5)
    photo(a, ims[0], 5.52, 3.28, 1.58, 1.63, H)
    arrow(a, 3.91, 4.12, 5.42, 4.12, ORANGE, 1.0, '--')
    txt(a, 4.65, 4.42, 'image', 8, ha='center', color=ORANGE)
    photo(a, ims[1], 8.73, 3.28, 1.55, 1.63, H)
    node(a, 11.8, 4.12, '$t_2$', NODE[1], 0.25, 9)
    arrow(a, 11.52, 4.12, 10.39, 4.12, ORANGE, 1.0, '--')
    txt(a, 10.95, 4.42, 'image', 8, ha='center', color=ORANGE)
    node(a, 14.35, 4.69, '$t_1$', NODE[0], 0.235, 8.8)
    node(a, 14.35, 3.85, '$t_3$', NODE[2], 0.235, 8.8)
    txt(a, 11.8, 3.43, 'True match', 9.0, True, ha='center')
    txt(a, 14.35, 3.32, 'Retrieved for $s_2$', 8.4, ha='center')
    a.plot([13.92, 13.72, 13.72, 13.92], [4.98, 4.98, 3.54, 3.54], color='#252525', lw=0.6, solid_joinstyle='miter')
    a.plot([3.62, 3.62, 11.8], [4.4, 5.55, 5.55], color=RED, lw=0.95, ls=':', dash_capstyle='round', solid_joinstyle='miter', zorder=2)
    arrow(a, 11.8, 5.55, 11.8, 4.4, RED, 0.95, ':')
    txt(a, 7.7, 5.74, 'Reference identity', 8.2, True, ha='center', color=RED)
    line(a, 0.1, 2.88, 15.9, 2.88, '#707070', 0.8, '--')
    line(a, 8.0, 0.08, 8.0, 2.65, '#707070', 0.8, '--')
    metrics = []
    for x, p, title, col in [(0, edge, '(b) Keep retrieved edges', BLUE), (8.25, pool, '(c) Pool candidate targets', ORANGE)]:
        txt(a, x + 0.1, 2.6, title, 9.5, True)
        for i, j in sorted(E):
            line(a, x + 0.58, 2.04 - i * 0.64, x + 3.25, 2.04 - j * 0.64, '#C3C6CA', 0.75)
        for i, j in enumerate(p):
            line(a, x + 0.58, 2.04 - i * 0.64, x + 3.25, 2.04 - j * 0.64, col, 1.6, '--' if (i, int(j)) not in E else '-', 3)
        for i in range(3):
            node(a, x + 0.58, 2.04 - i * 0.64, f'$s_{i + 1}$', NODE[i], 0.23, 8.3)
            node(a, x + 3.25, 2.04 - i * 0.64, f'$t_{i + 1}$', NODE[i], 0.23, 8.3)
        txt(a, x + 4.3, 2.05, 'Query', 8.5, True)
        txt(a, x + 7.0, 2.05, 'F1', 8.5, True, ha='center')
        for y, u, label in [(1.47, {1, 2}, 'Broad $U_b$'), (0.87, {1}, 'Narrow $U_n$')]:
            ans = {int(p[i]) for i in u}
            score = 2 * len(ans & u) / (len(ans) + len(u))
            metrics.append({'contract': title, 'source_group': sorted(u), 'answer': sorted(ans), 'f1': score})
            txt(a, x + 4.3, y, label, 8.5)
            txt(a, x + 7.0, y, f'{score:.0%}', 9.6, True, ha='center')
        txt(a, x + 1.92, 0.2, f'Correct identities: {sum(p == np.arange(3))}/3', 8.4, ha='center')
        txt(a, x + 5.65, 0.2, 'Unique targets: 3/3', 8.4, ha='center')
    out = save(f, 'motivation', output_dir)
    audit = {'revision': 'candidate_interface_study', 'synthetic_explanatory_example': True, 'scores': S.tolist(), 'edges': C.tolist(), 'gold': [0, 1, 2], 'edge_prediction': edge.tolist(), 'pool_prediction': pool.tolist(), 'new_pool_edge': [1, 1], 'query_metrics': metrics, 'enumerated_all_six_permutations': True, 'images_are_public_identity_illustrations': True, 'image_scores_are_not_measured_from_displayed_photographs': True, 'solid_closed_arrowheads': True, 'retrieved_targets_for_s2': [0, 2], 'true_target_for_s2': 1, 'target_nodes_are_distinct_entities_not_a_relation_chain': True, 'lower_panels_use_full_width': True, 'confirmed_layout_applied': True, 't2_caption_centered_below_shifted_node': True, 'reference_arrow_final_segment_vertical': True, 'target_entities_heading_covers_all_three_nodes': True}
    (out / 'motivation_audit.json').write_text(json.dumps(audit, indent=2))
    (R / 'visual').mkdir(exist_ok=True)
    (R / 'visual/motivation_check.json').write_text(json.dumps(audit, indent=2))
    return audit

def draw_case(output_dir=None):
    case = json.loads((R / 'visual/real_case.json').read_text())
    ims = images()
    assert case['source_id'] == 9881 and len(case['records']) == 12
    lookup = {(x['method'], x['seed']): x for x in case['records']}
    x = lookup['visual_zero_label_k16_union', 1]
    assert x['target_id'] == 13013 and (not x['prediction_visual_edge'])
    H = 5.12
    f, a = canvas(H)
    xs = [0.1, 3.05, 6.0, 8.95, 12.74, 14.28, 15.84]
    top = 4.94
    mid = 4.51
    headerbottom = 4.0
    bottom = 0.19
    a.add_patch(Rectangle((xs[0], headerbottom), xs[-1] - xs[0], top - headerbottom, fc='#F1F1F1', ec='none'))
    for xx in xs:
        line(a, xx, bottom, xx, mid if xx == xs[5] else top, '#656565', 0.55, '--')
    for yy in [top, headerbottom, bottom]:
        line(a, xs[0], yy, xs[-1], yy, '#656565', 0.65, '--')
    for l, r, title in [(0, 1, 'Source image'), (1, 2, 'Reference image'), (2, 3, 'Seed-1 pool error'), (3, 4, 'Decoder')]:
        txt(a, (xs[l] + xs[r]) / 2, 4.47, title, 9.0, True, ha='center')
    txt(a, (xs[4] + xs[6]) / 2, 4.72, 'Correct prediction', 8.5, True, ha='center')
    line(a, xs[4], mid, xs[6], mid, '#656565', 0.55, '--')
    for l, r, label in [(4, 5, 'Seed 0'), (5, 6, 'Seed 1')]:
        txt(a, (xs[l] + xs[r]) / 2, 4.24, label, 8.6, True, ha='center')
    for j, (im, name) in enumerate(zip(ims, ['Frédéric VIII', 'Frederick VIII', 'Haakon VII'])):
        photo(a, im, xs[j] + 0.21, 0.88, xs[j + 1] - xs[j] - 0.42, 2.77, H)
        txt(a, (xs[j] + xs[j + 1]) / 2, 0.56, name, 8.6, True, ha='center')
    rh = (headerbottom - bottom) / 6
    for k, (m, label) in enumerate(METHODS):
        y = headerbottom - rh * (k + 0.5)
        if k:
            line(a, xs[3], headerbottom - rh * k, xs[6], headerbottom - rh * k, '#929292', 0.45, '--')
        txt(a, (xs[3] + xs[4]) / 2, y, label, 8.9, ha='center')
        for seed, l, r in [(0, 4, 5), (1, 5, 6)]:
            ok = lookup[m, seed]['correct']
            txt(a, (xs[l] + xs[r]) / 2, y, 'Yes' if ok else 'No', 8.9, ha='center', bold=True, color='black')
    a.add_patch(Rectangle((xs[5] + 0.13, bottom + 0.1), xs[6] - xs[5] - 0.26, rh - 0.2, fc='none', ec=ORANGE, lw=1.0))
    out = save(f, 'real_case', output_dir)
    audit = {'revision': 'decision_case', 'source_case_id': 9881, 'existing_case_records_unchanged': True, 'all_12_decisions_displayed': True, 'displayed_image_sha256': [h for _, h in MEMBERS], 'error_image_corresponds_only_to_seed1_visual_pool': True, 'image_to_cached_feature_pixel_identity': 'unverified; public entity illustrations', 'records': case['records']}
    (out / 'real_case_audit.json').write_text(json.dumps(audit, indent=2))
    return audit
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir')
    args = p.parse_args()
    draw_motivation(args.output_dir)
    draw_case(args.output_dir)
