import json, time, hashlib
from pathlib import Path
import numpy as np
from confirmatory_data import workload
from confirmatory_eval import answers
from confirmatory_sql import connect, sql
R = Path(__file__).resolve().parent
O = R / 'candidate_continuation'
OUT = O / 'sql'
OUT.mkdir(exist_ok=True)

def main():
    records = []
    for mf in sorted(O.glob('*_manifest.json')):
        m = json.loads(mf.read_text())
        ds = m['dataset']
        left = np.array(m['sources'])
        right = np.array(m['nested_target_order'])
        w = workload(ds, left, right, right)
        db = connect(ds)
        files = list(O.glob('*' + ds.replace('/', '_') + '_block0_initial_results.json'))
        assert len(files) == 1, files
        f = files[0]
        preds = np.load(f.with_suffix('.npz'))
        count = 0
        tick = time.perf_counter()
        checks = []
        for size in m['sizes']:
            order = np.random.default_rng(29002 + size).permutation(size)
            targets = right[order]
            for depth in ['depth0', 'depth2']:
                for name in ['row', 'dense', 'stable', 'sparse']:
                    p = preds[f'{depth}_{size}_{name}']
                    mapping = [(int(a), int(targets[j])) for a, j in zip(left, p)]
                    db.execute('DELETE FROM mapping')
                    db.executemany('INSERT INTO mapping VALUES(?,?)', mapping)
                    db.execute('DELETE FROM unique_mapping')
                    accepted = True
                    try:
                        db.executemany('INSERT INTO unique_mapping VALUES(?,?)', mapping)
                    except Exception as e:
                        import sqlite3
                        if not isinstance(e, sqlite3.IntegrityError):
                            raise
                        accepted = False
                    assert accepted == (len(np.unique(p)) == len(p))
                    checks.append(dict(targets=size, score=depth, decoder=name, unique_target_key=accepted))
                    for q, (U, V, keys) in w.items():
                        expected = answers(p, U, size) & V[:, order]
                        for j, key in enumerate(keys):
                            query, params = sql(q, key)
                            observed = {row[0] for row in db.execute(query, params)}
                            want = set(map(int, targets[expected[j]]))
                            assert observed == want, (ds, size, depth, name, q, j)
                            count += 1
        db.close()
        rec = dict(dataset=ds, model_seed=0, mode='initial', executed_queries=count, exact_set_equality=True, checks=checks, source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(), elapsed_s=time.perf_counter() - tick)
        records.append(rec)
        (OUT / (ds.replace('/', '_') + '.json')).write_text(json.dumps(rec, indent=2))
        print('SQL candidate_continuation ' + ds + ' ' + str(count), flush=True)
    (OUT / 'summary.json').write_text(json.dumps(dict(records=records, executed_queries=sum((r['executed_queries'] for r in records)), exact_set_equality=True), indent=2))
if __name__ == '__main__':
    main()
