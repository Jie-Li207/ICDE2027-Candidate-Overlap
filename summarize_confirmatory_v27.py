import json, collections
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent
O = R / 'confirmatory_v27'

def bootstrap(matrix):
    matrix = np.asarray(matrix, float)
    rng = np.random.default_rng(27001)
    s, b = matrix.shape
    si = rng.integers(s, size=(10000, s))
    bi = rng.integers(b, size=(10000, b))
    sample = matrix[si[:, :, None], bi[:, None, :]].mean((1, 2))
    return dict(mean=float(matrix.mean()), lower=float(np.quantile(sample, 0.025)), upper=float(np.quantile(sample, 0.975)), model_seeds=s, data_blocks=b, interval_scope='conditional benchmark crossed seed/block bootstrap' if s > 1 else 'conditional on this model, data-block bootstrap')

def main():
    rows = []
    for file in sorted((O / 'results').glob('*.json')):
        try:
            rows.extend(json.loads(file.read_text())['rows'])
        except json.JSONDecodeError:
            continue
    (O / 'all_results.json').write_text(json.dumps(rows))
    group = collections.defaultdict(list)
    dims = ['dataset', 'learner', 'mode', 'regime', 'score', 'decoder', 'query_set']
    for r in rows:
        group[tuple((r[k] for k in dims))].append(r)
    summary = []
    for k, rs in group.items():
        seeds = sorted(set((r['seed'] for r in rs)))
        matrix = {(r['seed'], r['block']): r for r in rs}
        if not all(((s, b) in matrix for s in seeds for b in range(4))):
            continue
        out = dict(zip(dims, k))
        out['seeds'] = seeds
        for metric in ['entity_accuracy', 'target_coverage', 'micro_f1', 'macro_f1', 'precision', 'recall']:
            out[metric] = bootstrap([[matrix[s, b][metric] for b in range(4)] for s in seeds])
        summary.append(out)
    paired = []
    basegroup = collections.defaultdict(dict)
    for r in rows:
        key = tuple((r[k] for k in ['dataset', 'learner', 'mode', 'regime', 'score', 'query_set']))
        basegroup[key][r['seed'], r['block'], r['decoder']] = r
    for key, rs in basegroup.items():
        for a, b in [('row', 'dense'), ('ot15_row', 'ot15_dense'), ('ot500_row', 'ot500_dense'), ('dense', 'sparse'), ('dense', 'stable')]:
            seeds = sorted(set((s for s, block, dec in rs if dec == a and (s, block, b) in rs)))
            if not seeds or not all(((s, j, d) in rs for s in seeds for j in range(4) for d in [a, b])):
                continue
            rec = dict(zip(['dataset', 'learner', 'mode', 'regime', 'score', 'query_set'], key))
            rec.update(first=a, second=b, seeds=seeds)
            for metric in ['entity_accuracy', 'micro_f1', 'macro_f1', 'target_coverage']:
                rec[metric + '_delta'] = bootstrap([[rs[s, j, b][metric] - rs[s, j, a][metric] for j in range(4)] for s in seeds])
            rec['rank_reversal_blocks'] = sum(((rs[s, j, b]['entity_accuracy'] - rs[s, j, a]['entity_accuracy']) * (rs[s, j, b]['micro_f1'] - rs[s, j, a]['micro_f1']) < 0 for s in seeds for j in range(4)))
            paired.append(rec)
    (O / 'summary.json').write_text(json.dumps(summary, indent=2))
    (O / 'paired.json').write_text(json.dumps(paired, indent=2))
    selected = [r for r in paired if r['learner'] == 'DESAlign' and r['mode'] == 'initial' and (r['score'] == 'depth2') and (r['query_set'] == 'coarse') and (r['first'] in ['row', 'ot500_row'])]
    print(json.dumps(selected, indent=2))
    print('ROWS ' + str(len(rows)) + ' COMPLETE GROUPS ' + str(len(summary)))
if __name__ == '__main__':
    main()
