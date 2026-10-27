import sqlite3, json
from pathlib import Path
import numpy as np
from confirmatory_data_v27 import b
R = Path(__file__).resolve().parent
O = R / 'revision_v34'
checked = 0
per = []
for ds in ['FBDB15K/norm', 'FBYG15K/norm', 'DBP15K/zh_en', 'DBP15K/fr_en']:
    con = sqlite3.connect(':memory:')
    for side in [1, 2]:
        con.execute(f'CREATE TABLE f{side}(h INTEGER,r INTEGER,t INTEGER)')
        triples = np.loadtxt(b.BASE / ds / f'triples_{side}', dtype=np.int64)
        con.executemany(f'INSERT INTO f{side} VALUES(?,?,?)', ((int(h), int(r), int(t)) for h, r, t in triples))
        con.execute(f'CREATE INDEX i{side} ON f{side}(r,h,t)')
        con.execute(f'CREATE INDEX j{side} ON f{side}(h,r,t)')
    for cohort in ['diagnostic256', 'fresh64']:
        stem = ds.replace('/', '_') + '_' + cohort
        wm = json.loads((O / (stem + '_workload.json')).read_text())
        z = np.load(O / (stem + '_workload.npz'))
        n = 0
        for q, x in enumerate(wm['queries']):
            family = x['family']
            key = x['key']
            v = None
            if family in ['relation', 'original_coarse']:
                sql = 'SELECT DISTINCT h FROM f1 WHERE r=?'
                args = (key,)
            elif family in ['relation_object', 'original_value']:
                sql = 'SELECT DISTINCT h FROM f1 WHERE r=? AND t=?'
                args = tuple(key)
            elif family == 'two_hop':
                sql = 'SELECT DISTINCT a.h FROM f1 a JOIN f1 b ON a.t=b.h WHERE a.r=? AND b.r=?'
                args = tuple(key)
            elif family == 'original_cross_filter':
                sql = 'SELECT DISTINCT h FROM f1 WHERE r=?'
                args = (key[0],)
                v = {r[0] for r in con.execute('SELECT DISTINCT h FROM f2 WHERE r=?', (key[1],))}
            else:
                raise ValueError(family)
            u = {r[0] for r in con.execute(sql, args)}
            assert np.array_equal(z['U'][q], [int(i) in u for i in z['left']])
            assert np.array_equal(z['V'][q], np.ones(len(z['right']), bool) if v is None else [int(i) in v for i in z['right']])
            checked += 1
            n += 1
        per.append(dict(dataset=ds, cohort=cohort, queries=n))
        print(ds, cohort, n, flush=True)
    con.close()
(O / 'raw_fact_sql_validation.json').write_text(json.dumps(dict(passed=True, queries=checked, cases=per, scope='Every saved KG source and target predicate derived independently from original triple tables in SQLite, including two-hop joins.'), indent=2))
