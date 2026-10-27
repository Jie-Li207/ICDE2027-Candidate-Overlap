import json, sqlite3, hashlib
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent

def run(folder):
    O = R / folder
    metric_rows = conditions = sql_queries = 0
    fail = []
    files = sorted(O.glob('*_results.json'))
    for path in files:
        data = json.loads(path.read_text())
        first = data['rows'][0]
        if folder in ['revision_v34', 'revision_v39', 'revision_v41']:
            stem = first['dataset'].replace('/', '_') + '_' + first['cohort']
            do_sql = first['seed'] == 0
        else:
            stem = first['dataset'] + f"_block{first['block']}"
            do_sql = True
        z = np.load(O / (stem + '_workload.npz'))
        wm = json.loads((O / (stem + '_workload.json')).read_text())
        pred = np.load(path.with_suffix('.npz'))
        U = z['U']
        V = z['V']
        gold = z['gold']
        n = len(gold)
        G = []
        conn = sqlite3.connect(':memory:')
        conn.executescript('CREATE TABLE member(q INTEGER,s INTEGER);CREATE INDEX mi ON member(q,s);CREATE TABLE mapping(s INTEGER PRIMARY KEY,t INTEGER);CREATE TABLE filter(q INTEGER,t INTEGER);CREATE INDEX fi ON filter(q,t);')
        if do_sql:
            conn.executemany('INSERT INTO member VALUES(?,?)', [(q, int(i)) for q, u in enumerate(U) for i in np.flatnonzero(u)])
            conn.executemany('INSERT INTO filter VALUES(?,?)', [(q, int(j)) for q, v in enumerate(V) if not v.all() for j in np.flatnonzero(v)])
        for q, u in enumerate(U):
            G.append({int(gold[i]) for i in np.flatnonzero(u) if V[q, gold[i]]})
        for c in data['conditions']:
            cid = c['id']
            p = pred[cid + '_p']
            assert np.all((p >= -1) & (p < V.shape[1]))
            counts = []
            if c['decoder'] == 'edge':
                assert len(set(p[p >= 0])) == np.sum(p >= 0)
            if c['retriever'] not in ['full', 'oracle_closed']:
                C = pred[c['retriever'] + f"_k{c['k']}_edges"]
                assert abs(np.mean((C == gold[:, None]).any(1)) - c['edge_recall']) < 1e-12
                if c['decoder'] in ['edge', 'row']:
                    assert all((p[i] < 0 or p[i] in C[i] for i in range(n)))
                if c['decoder'] == 'union':
                    assert abs(c['unsupported_edge_fraction'] - np.mean((p >= 0) & ~(C == p[:, None]).any(1))) < 1e-12
            if do_sql:
                conn.execute('DELETE FROM mapping')
                conn.executemany('INSERT INTO mapping VALUES(?,?)', [(i, int(v)) for i, v in enumerate(p) if v >= 0])
            for q, u in enumerate(U):
                A = {int(p[i]) for i in np.flatnonzero(u) if p[i] >= 0 and V[q, p[i]]}
                g = G[q]
                counts.append([len(A & g), len(A - g), len(g - A)])
                if do_sql:
                    sql = 'SELECT DISTINCT m.t FROM mapping m JOIN member u ON u.s=m.s WHERE u.q=?'
                    args = (q,)
                    if not V[q].all():
                        sql += ' AND EXISTS(SELECT 1 FROM filter f WHERE f.q=? AND f.t=m.t)'
                        args = (q, q)
                    observed = {x[0] for x in conn.execute(sql, args)}
                    assert observed == A
                    sql_queries += 1
            counts = np.array(counts)
            assert np.array_equal(counts, pred[cid + '_counts'])
            conditions += 1
            for row in [x for x in data['rows'] if x['id'] == cid]:
                ix = [i for i, q in enumerate(wm['queries']) if row['family'] == 'all' or (q['family'] == row['family'] and str(q['bin']) == row['bin'])]
                tp, fp, fn = counts[ix].sum(0)
                assert [row['TP'], row['FP'], row['FN']] == [tp, fp, fn]
                expected = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
                assert abs(expected - row['micro_f1']) < 1e-12
                assert abs(row['entity_accuracy'] - np.mean(p == gold)) < 1e-12
                metric_rows += 1
        conn.close()
        print('VERIFIED ' + path.stem, flush=True)
    result = dict(passed=True, case_files=len(files), conditions=conditions, metric_rows=metric_rows, sql_queries=sql_queries, scope='Independent sets recompute every recorded condition; actual SQLite materialized-membership joins validate seed0 KG cases and every product case. This SQL check does not claim independently deriving two-hop membership from original triple tables.')
    (O / 'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
if __name__ == '__main__':
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else 'revision_v34')
