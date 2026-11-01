import json, collections, sqlite3
import numpy as np
from retrieved_candidates import R, BINS, digest, dump, per_query, summarize
from confirmatory_data import b
O = R / 'twohop_queries'
O.mkdir(exist_ok=True)
rawchecks = 0
for ds in ['FBDB15K/norm', 'FBYG15K/norm', 'DBP15K/zh_en', 'DBP15K/fr_en']:
    T = np.loadtxt(b.BASE / ds / 'triples_1', dtype=np.int64)
    adj = collections.defaultdict(list)
    for h, r, t in T:
        adj[int(h)].append((int(r), int(t)))
    con = sqlite3.connect(':memory:')
    con.execute('CREATE TABLE f(h INTEGER,r INTEGER,t INTEGER)')
    con.executemany('INSERT INTO f VALUES(?,?,?)', ((int(h), int(r), int(t)) for h, r, t in T))
    con.execute('CREATE INDEX fhr ON f(h,r,t)')
    con.execute('CREATE INDEX frh ON f(r,h,t)')
    for cohort in ['diagnostic256', 'fresh64']:
        stem = ds.replace('/', '_') + '_' + cohort
        old = np.load(R / 'retrieved_candidates' / (stem + '_workload.npz'))
        left = old['left']
        pred = collections.defaultdict(set)
        for h in left:
            for r, x in adj[int(h)]:
                for s, y in adj[x]:
                    pred[r, s].add(int(h))
        U = []
        meta = []
        counts = collections.Counter()
        seen = set()
        for key in sorted(pred, key=lambda x: digest(json.dumps(x).encode())):
            u = np.array([int(h) in pred[key] for h in left])
            n = int(u.sum())
            bi = int(np.searchsorted(BINS, n / len(left), side='left') - 1)
            if u.tobytes() in seen:
                continue
            seen.add(u.tobytes())
            if counts[bi] >= 8:
                continue
            counts[bi] += 1
            U.append(u)
            meta.append(dict(family='two_hop', bin=bi, key=key, source_count=n))
            got = {r[0] for r in con.execute('SELECT DISTINCT a.h FROM f a JOIN f b ON a.t=b.h WHERE a.r=? AND b.r=?', key)}
            assert np.array_equal(u, [int(h) in got for h in left])
            rawchecks += 1
        assert U, 'No actual two-hop predicates'
        U = np.array(U)
        V = np.ones((len(U), len(old['right'])), bool)
        wm = dict(dataset=ds, cohort=cohort, queries=meta, query_count=len(U), bin_edges=BINS, actual_path_relation_pairs=len(pred), selection='Hash order over observed paths, at most8 distinct memberships per selectivity bin', protocol_sha256=digest((R / 'protocol_twohop.json').read_bytes()))
        np.savez_compressed(O / (stem + '_workload.npz'), left=left, right=old['right'], gold=old['gold'], U=U, V=V)
        dump(O / (stem + '_workload.json'), wm)
        for p in sorted((R / 'retrieved_candidates').glob(ds.replace('/', '_') + '_seed*_' + cohort + '_results.json')):
            d = json.loads(p.read_text())
            z = np.load(p.with_suffix('.npz'))
            arrays = {k: z[k] for k in z.files if k.endswith('_p') or k.endswith('_edges')}
            rows = []
            for c in d['conditions']:
                cid = c['id']
                predictions = arrays[cid + '_p']
                cnt = per_query(predictions, U, V, old['gold'])
                arrays[cid + '_counts'] = cnt
                rows.extend((dict(dataset=ds, seed=d['seed'], cohort=cohort, **c, **x) for x in summarize(predictions, cnt, wm, old['gold'])))
            np.savez_compressed(O / p.with_suffix('.npz').name, **arrays)
            dump(O / p.name, dict(dataset=ds, seed=d['seed'], cohort=cohort, conditions=d['conditions'], rows=rows, prediction_source_sha256=digest(p.with_suffix('.npz').read_bytes()), protocol_sha256=wm['protocol_sha256'], producer_sha256=digest(__import__('pathlib').Path(__file__).read_bytes())))
        print(ds, cohort, 'queries', len(U), 'bins', dict(counts), flush=True)
    con.close()
dump(O / 'raw_fact_sql_validation.json', dict(passed=True, queries=rawchecks, scope='Every new two-hop membership independently derived using joins over original triples; predictions are unchanged from retrieved_candidates.'))
