import collections
import csv
import json
from pathlib import Path
import numpy as np
from query_scope import R, O, P, dump

def crossed(values, retail=False):
    seeds = sorted({v[0] for v in values})
    blocks = sorted({v[1] for v in values})
    lookup = {(s, b): v for s, b, v in values}
    A = np.array([[lookup[s, b] for b in blocks] for s in seeds])
    rng = np.random.default_rng(53001)
    B = rng.integers(len(blocks), size=(10000, len(blocks)))
    if retail:
        samples = A.mean(0)[B].mean(1)
    else:
        I = rng.integers(len(seeds), size=(10000, len(seeds)))
        samples = A[I[:, :, None], B[:, None, :]].mean((1, 2))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return dict(mean=float(A.mean()), lo=float(lo), hi=float(hi), models=len(seeds), blocks=len(blocks), conditional=True)

def main():
    data = []
    for p in O.glob('*_k*.json'):
        r = json.loads(p.read_text())
        if 'plans' in r:
            data.append(r)
    groups = collections.defaultdict(list)
    for r in data:
        groups[r['dataset'], r['scorer'], r['retriever'], r['k']].append(r)
    aggregates = []
    for (ds, scorer, retriever, k), rows in sorted(groups.items()):
        q = dict(dataset=ds, scorer=scorer, retriever=retriever, k=k)
        metrics = {}
        for plan in rows[0]['plans']:
            for field in ['changed_fraction', 'decision_flip_fraction', 'query_f1', 'baseline_f1', 'delta_f1']:
                metrics[plan + '_' + field] = crossed([(r['seed'], r['block'], r['plans'][plan][field]) for r in rows], ds == 'RetailVisualRAG')
        for field in ['edge_recall', 'pool_recall', 'mean_closure_fraction']:
            metrics[field] = crossed([(r['seed'], r['block'], r['graph'][field]) for r in rows], ds == 'RetailVisualRAG')
        for name in ['row', 'edge', 'pool']:
            for field in ['identity_accuracy', 'query_f1']:
                metrics[name + '_' + field] = crossed([(r['seed'], r['block'], r['global_metrics'][name][field]) for r in rows], ds == 'RetailVisualRAG')
        metrics['pool_minus_edge_f1'] = crossed([(r['seed'], r['block'], r['global_metrics']['pool']['query_f1'] - r['global_metrics']['edge']['query_f1']) for r in rows], ds == 'RetailVisualRAG')
        q['metrics'] = metrics
        q['conditions'] = len(rows)
        q['nonempty_queries_per_condition'] = sorted({r['plans']['pool_local']['nonempty_queries'] for r in rows})
        aggregates.append(q)
    dump(O / 'aggregate.json', aggregates)
    primary = []
    for ds in json.loads(P.read_text())['graph_cohorts']:
        checks = []
        for ret in ['cosine', 'csls10']:
            a = next((x for x in aggregates if x['dataset'] == ds and x['retriever'] == ret and (x['k'] == 16)))
            checks.append(dict(retriever=ret, edge=a['metrics']['edge_local_changed_fraction'], pool=a['metrics']['pool_local_changed_fraction'], passes=a['metrics']['pool_local_changed_fraction']['mean'] >= 0.05))
        primary.append(dict(dataset=ds, checks=checks, passes=all((x['passes'] for x in checks))))
    retail = []
    for a in aggregates:
        if a['dataset'] == 'RetailVisualRAG' and a['scorer'] == 'ridge128' and (a['k'] == 4):
            delta = a['metrics']['pool_minus_edge_f1']
            change = a['metrics']['pool_local_changed_fraction']
            retail.append(dict(retriever=a['retriever'], pool_minus_edge_f1=delta, changed=change, passes=abs(delta['mean']) >= 0.01 or change['mean'] >= 0.05))
    dump(O / 'hypothesis_checks.json', dict(graph=primary, graph_primary_pass=sum((x['passes'] for x in primary)) >= 2, retail=retail, retail_replication_pass=all((x['passes'] for x in retail)) if retail else None, not_a_paper_acceptance_gate=True))
    rows = []
    for a in aggregates:
        for metric, value in a['metrics'].items():
            rows.append({k: v for k, v in a.items() if k not in ['metrics', 'nonempty_queries_per_condition']} | dict(metric=metric, **value))
    with (O / 'aggregate.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('AGGREGATED', len(data), 'conditions', len(aggregates), 'groups', flush=True)
    for a in aggregates:
        if a['k'] == (4 if a['dataset'] == 'RetailVisualRAG' else 16):
            print(a['dataset'], a['scorer'], a['retriever'], {key: round(a['metrics'][key]['mean'] * 100, 2) for key in ['edge_local_changed_fraction', 'pool_local_changed_fraction', 'pool_fixed_changed_fraction', 'mean_closure_fraction', 'pool_minus_edge_f1']}, flush=True)
if __name__ == '__main__':
    main()
