import json, sqlite3, time, hashlib
from pathlib import Path
import numpy as np
from confirmatory_eval_v27 import R, O, RESULT, answers
BASE = R / 'public_data/desalign/selected/desalign_datasets'
OUT = O / 'sql'
OUT.mkdir(exist_ok=True)

def connect(ds):
    file = OUT / (ds.replace('/', '_') + '.sqlite')
    new = not file.exists()
    db = sqlite3.connect(file)
    if new:
        for side, name in [(1, 'source_fact'), (2, 'target_fact')]:
            db.execute(f'CREATE TABLE {name}(h INTEGER,r INTEGER,t INTEGER)')
            T = np.loadtxt(BASE / ds / f'triples_{side}', dtype=np.int64)
            db.executemany(f'INSERT INTO {name} VALUES(?,?,?)', ((int(h), int(r), int(t)) for h, r, t in T))
            db.execute(f'CREATE INDEX {name}_hrt ON {name}(h,r,t)')
            db.execute(f'CREATE INDEX {name}_rth ON {name}(r,t,h)')
        db.commit()
    db.execute('CREATE TEMP TABLE mapping(s INTEGER PRIMARY KEY,t INTEGER NOT NULL)')
    db.execute('CREATE INDEX mapping_t ON mapping(t)')
    db.execute('CREATE TEMP TABLE unique_mapping(s INTEGER PRIMARY KEY,t INTEGER NOT NULL UNIQUE)')
    return db

def sql(q, key):
    if q == 'coarse':
        return ('SELECT DISTINCT m.t FROM mapping m JOIN source_fact f ON f.h=m.s WHERE f.r=?', (key,))
    if q == 'value':
        return ('SELECT DISTINCT m.t FROM mapping m JOIN source_fact f ON f.h=m.s WHERE f.r=? AND f.t=?', tuple(key))
    return ('SELECT DISTINCT m.t FROM mapping m JOIN source_fact f ON f.h=m.s JOIN target_fact g ON g.h=m.t WHERE f.r=? AND g.r=?', tuple(key))

def main():
    dbs = {}
    records = []
    for file in sorted(RESULT.glob('*.json')):
        content = json.loads(file.read_text())
        row = content['rows'][0]
        ds = row['dataset']
        block = row['block']
        regime = row['regime']
        if row['seed'] != 0 or block != 0 or row['mode'] not in ['initial', 'noimage']:
            continue
        if row['learner'] not in ['DESAlign', 'DESAlign+TFP', 'CDMEA-50', 'CDMEA-200']:
            continue
        if row['score'] not in ['depth2', 'tfp_cosine', 'CDI_CSLS']:
            continue
        target = OUT / (file.stem + '.json')
        if target.exists():
            records.append(json.loads(target.read_text()))
            continue
        if ds not in dbs:
            dbs[ds] = connect(ds)
        db = dbs[ds]
        z = np.load(O / (ds.replace('/', '_') + f'_block{block}_{regime}_workload.npz'))
        queries = json.loads((O / (ds.replace('/', '_') + f'_block{block}_{regime}_queries.json')).read_text())
        pred = np.load(file.with_suffix('.npz'))
        left = z['left']
        right = z['right']
        count = 0
        keys = []
        tick = time.perf_counter()
        for name in list(pred.files) + ['reference_gold']:
            p = z['gold'] if name == 'reference_gold' else pred[name]
            mapping = [(int(a), int(right[j])) for a, j in zip(left, p)]
            db.execute('DELETE FROM mapping')
            db.executemany('INSERT INTO mapping VALUES(?,?)', mapping)
            db.execute('DELETE FROM unique_mapping')
            accepted = True
            try:
                db.executemany('INSERT INTO unique_mapping VALUES(?,?)', mapping)
            except sqlite3.IntegrityError:
                accepted = False
            assert accepted == (len(np.unique(p)) == len(p))
            keys.append(dict(decoder=name, key_accepted=accepted))
            for q in ['coarse', 'value', 'cross_filter']:
                expected = answers(p, z[q + '_U'], len(right)) & z[q + '_V']
                for j, key in enumerate(queries[q]):
                    query, params = sql(q, key)
                    observed = {r[0] for r in db.execute(query, params)}
                    want = set(map(int, right[expected[j]]))
                    assert observed == want, (file.name, name, q, j, len(observed), len(want))
                    count += 1
        rec = dict(result=file.name, source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(), executed_queries=count, exact_set_equality=True, key_checks=keys, elapsed_s=time.perf_counter() - tick)
        target.write_text(json.dumps(rec, indent=2))
        records.append(rec)
        print('SQL ' + file.stem + ' ' + str(count), flush=True)
    for db in dbs.values():
        db.close()
    (OUT / 'summary.json').write_text(json.dumps(dict(conditions=len(records), executed_queries=sum((r['executed_queries'] for r in records)), records=records), indent=2))
    print('SQL COMPLETE', flush=True)
if __name__ == '__main__':
    main()
