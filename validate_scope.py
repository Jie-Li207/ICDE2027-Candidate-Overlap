import ast
import json
import sqlite3
from pathlib import Path
import numpy as np
import torch
from query_scope import R, O, P, sha, dump, answer, counts, partial, edge

def validate_csls():
    p = R / 'thirdparty' / 'DESAlign' / 'src' / 'utils.py'
    tree = ast.parse(p.read_text(encoding='utf8'))
    fn = next((x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'csls_sim'))
    env = {'torch': torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(p), 'exec'), env)
    a = np.random.default_rng(53119).normal(size=(31, 47)).astype(np.float32)
    expected = env['csls_sim'](torch.from_numpy(a), 10).numpy()
    row = np.partition(a, a.shape[1] - 10, axis=1)[:, -10:].mean(1)
    col = np.partition(a, a.shape[0] - 10, axis=0)[-10:].mean(0)
    actual = 2 * a - row[:, None] - col[None, :]
    error = float(abs(expected - actual).max())
    assert error < 1e-06
    return dict(upstream='DESAlign', file_sha256=sha(p.read_bytes()), maximum_error=error, passed=True)

def main():
    results = []
    for p in sorted(O.glob('*_k*.json')):
        a = json.loads(p.read_text())
        if 'plans' in a:
            results.append((p, a))
    totals = dict(conditions=0, query_plan_checks=0, sql_query_checks=0, raw_fact_memberships=0, zero_control_checks=0)
    sqlcases = []
    for path, r in results:
        with np.load(path.with_suffix('.npz')) as source:
            z = {k: source[k] for k in source.files}
        C = z['C']
        U = z['U']
        gold = z['gold']
        n = len(gold)
        assert r['protocol_sha256'] == sha(P.read_bytes())
        for base in ['edge', 'pool']:
            v = z[base]
            assert len(set(v[v >= 0])) == (v >= 0).sum()
        assert all((z['edge'][i] in C[i] for i in range(n) if z['edge'][i] >= 0))
        totals['conditions'] += 1
        selected = r['dataset'] == 'RetailVisualRAG' and r['k'] == 4 or (r['dataset'] != 'RetailVisualRAG' and r['seed'] == 0 and (r['k'] == 16))
        if selected:
            conn = sqlite3.connect(':memory:')
            conn.executescript('CREATE TABLE mapping(s INTEGER PRIMARY KEY,t INTEGER);CREATE TABLE chosen(s INTEGER PRIMARY KEY);CREATE TABLE truth(s INTEGER PRIMARY KEY,t INTEGER);')
            conn.executemany('INSERT INTO truth VALUES(?,?)', [(i, int(t)) for i, t in enumerate(gold)])
        for plan, base in [('row_local', 'row'), ('edge_local', 'edge'), ('edge_closure', 'edge'), ('pool_local', 'pool'), ('pool_fixed', 'pool')]:
            actual = []
            baseline = []
            changed = []
            for q, u in enumerate(U):
                ix = np.flatnonzero(u)
                p = z[plan + '_pred'][q, ix]
                assert (p >= -1).all()
                if base in ['edge', 'pool']:
                    assert len(set(p[p >= 0])) == (p >= 0).sum()
                if base == 'edge':
                    assert all((p[j] in C[i] for j, i in enumerate(ix) if p[j] >= 0))
                c = counts(p, gold[ix])
                bc = counts(z[base][ix], gold[ix])
                assert np.array_equal(c, z[plan + '_counts'][q])
                assert np.array_equal(bc, z[plan + '_baseline_counts'][q])
                actual.append(c)
                baseline.append(bc)
                ch = answer(p) != answer(z[base][ix])
                changed.append(ch)
                assert ch == z[plan + '_changed'][q]
                if plan in ['row_local', 'edge_closure']:
                    assert not ch
                    totals['zero_control_checks'] += 1
                if selected:
                    conn.execute('DELETE FROM mapping')
                    conn.execute('DELETE FROM chosen')
                    conn.executemany('INSERT INTO mapping VALUES(?,?)', [(int(i), int(t)) for i, t in zip(ix, p) if t >= 0])
                    conn.executemany('INSERT INTO chosen VALUES(?)', [(int(i),) for i in ix])
                    ans = {v[0] for v in conn.execute('SELECT DISTINCT m.t FROM mapping m JOIN chosen u ON u.s=m.s')}
                    true = {v[0] for v in conn.execute('SELECT DISTINCT g.t FROM truth g JOIN chosen u ON u.s=g.s')}
                    assert ans == answer(p) and true == answer(gold[ix])
                    sqlc = [len(ans & true), len(ans - true), len(true - ans)]
                    assert sqlc == c
                    totals['sql_query_checks'] += 1
                totals['query_plan_checks'] += 1
            assert sum(changed) == r['plans'][plan]['changed_queries']
        if selected:
            conn.close()
            sqlcases.append(path.name)
    from confirmatory_data import b
    for ds in json.loads(P.read_text())['graph_cohorts']:
        tag = ds.replace('/', '_')
        conn = sqlite3.connect(':memory:')
        conn.execute('CREATE TABLE fact(h INTEGER,r INTEGER,t INTEGER)')
        triples = np.loadtxt(b.BASE / ds / 'triples_1', dtype=np.int64)
        conn.executemany('INSERT INTO fact VALUES(?,?,?)', [tuple(map(int, t)) for t in triples])
        conn.execute('CREATE INDEX fact_rh ON fact(r,h)')
        for block in range(4):
            mf = json.loads((O / (tag + f'_block{block}_manifest.json')).read_text())
            left = np.array(mf['sources'])
            z = np.load(O / (tag + f'_seed0_block{block}_cosine_k16.npz'))
            for q, x in enumerate(mf['query_keys']):
                key = x['key']
                if x['family'] == 'relation':
                    rows = conn.execute('SELECT DISTINCT h FROM fact WHERE r=?', (key,))
                elif x['family'] == 'value':
                    rows = conn.execute('SELECT DISTINCT h FROM fact WHERE r=? AND t=?', key)
                else:
                    rows = conn.execute('SELECT DISTINCT a.h FROM fact a JOIN fact c ON a.t=c.h WHERE a.r=? AND c.r=?', key)
                ids = {x[0] for x in rows}
                assert np.array_equal(np.array([int(v) in ids for v in left]), z['U'][q])
                totals['raw_fact_memberships'] += 1
        conn.close()
    if (O / 'retail_manifest.json').exists():
        import pandas as pd
        df = pd.read_parquet(R / 'public_data' / 'retail' / 'test.parquet')
        paths = ['test/' + str(a) + '/' + str(b) for a, b in zip(df['class'], df['filename'])]
        df['path'] = paths
        for view in [1, 2]:
            for block in range(4):
                mf = json.loads((O / f'Retail_view{view}_block{block}_manifest.json').read_text())
                data = df.set_index('path').loc[[mf['paths'][i] for i in mf['sources']]].copy()
                data['s'] = np.arange(len(data))
                conn = sqlite3.connect(':memory:')
                data.to_sql('offer', conn, index=False)
                z = np.load(O / f'Retail_view{view}_block{block}_ridge128_cosine_k4.npz')
                for q, x in enumerate(mf['queries']):
                    family = x['family']
                    if family == 'price':
                        rows = conn.execute('SELECT s FROM offer WHERE price<=?', (x['key'],))
                    elif family == 'discount':
                        rows = conn.execute('SELECT s FROM offer WHERE relative_discount IS NOT NULL OR absolute_discount IS NOT NULL')
                    elif family == 'different_sorts':
                        rows = conn.execute("SELECT s FROM offer WHERE lower(trim(different_sorts))='yes'")
                    elif family == 'brand':
                        rows = conn.execute('SELECT s FROM offer WHERE brand=?', (x['key'],))
                    else:
                        rows = conn.execute('SELECT s FROM offer WHERE product_category=?', (x['key'],))
                    ids = {i[0] for i in rows}
                    assert np.array_equal([i in ids for i in range(len(data))], z['U'][q])
                    totals['raw_fact_memberships'] += 1
                conn.close()
    report = dict(passed=True, **totals, csls=validate_csls(), sql_cases=sqlcases, protocol_sha256=sha(P.read_bytes()), scope='Recomputed every saved plan/query metric and selected SQL mapping execution; raw graph predicates checked against SQLite triple joins and retail predicates against source metadata. Scientific claims are assessed separately.')
    dump(O / 'validation.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'sql_cases'}, indent=2), flush=True)
if __name__ == '__main__':
    main()
