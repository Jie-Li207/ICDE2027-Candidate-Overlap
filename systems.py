import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, time, gc, argparse, ctypes, platform, sqlite3, psutil, hashlib
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from confirmatory_eval import R, O as DATA, assignment, stable, sparse, metrics
from confirmatory_data import workload
from confirmatory_sql import connect, sql
threadpool_limits(2)
O = R / 'systems'
O.mkdir(exist_ok=True)

class PMC(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [(s, ctypes.c_size_t) for s in ['PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]

def memory():
    p = PMC()
    p.cb = ctypes.sizeof(p)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    assert ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(p), p.cb)
    return dict(peak_working_set_bytes=p.PeakWorkingSetSize, working_set_bytes=p.WorkingSetSize, pagefile_bytes=p.PagefileUsage)

def repeat(fn):
    result = fn()
    times = []
    for _ in range(7):
        gc.collect()
        t = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t)
    return (result, dict(samples_s=times, median_s=float(np.median(times)), q25_s=float(np.quantile(times, 0.25)), q75_s=float(np.quantile(times, 0.75)), **memory()))

def ot15(S):
    P = S.astype(float) / 0.02
    for _ in range(15):
        P -= logsumexp(P, axis=1, keepdims=True)
        P -= logsumexp(P, axis=0, keepdims=True)
    return np.exp(P)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--expanded', action='store_true')
    args = parser.parse_args()
    ds = args.dataset
    n = args.n
    regime = 'expanded' if args.expanded else 'closed'
    stem = ds.replace('/', '_') + f'_{regime}_{n}'
    dest = O / (stem + '.json')
    if dest.exists():
        if json.loads(dest.read_text()).get('complete'):
            print('EXISTS ' + stem)
            return
        dest.with_suffix('.incomplete_' + str(int(time.time())) + '.json').write_bytes(dest.read_bytes())
    ancestors = {p.pid for p in psutil.Process().parents()} | {os.getpid()}
    while True:
        busy = []
        for process in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
            x = process.info
            if x['pid'] in ancestors or not (x['name'] or '').lower().startswith('python'):
                continue
            if x['cwd'] and Path(x['cwd']).resolve() == R and any((Path(c).name.startswith(('confirmatory_', 'candidate_controls_', 'cdmea_', 'desalign_', 'export_frozen_', 'tfp_')) and c.endswith('.py') for c in x['cmdline'] or [])):
                busy.append(x['pid'])
        if not busy:
            break
        time.sleep(15)
    man = json.loads((DATA / (ds.replace('/', '_') + '_identities.json')).read_text())
    pairs = np.array(man['test'][:n])
    assert len(pairs) == n
    left = pairs[:, 0]
    if args.expanded:
        right = np.array(man['gallery'])
        lookup = {i: j for j, i in enumerate(right)}
        gold = np.array([lookup[i] for i in pairs[:, 1]])
    else:
        order = np.random.default_rng(28001 + n).permutation(n)
        right = pairs[order, 1]
        gold = np.argsort(order)
    folder = 'desalign_gpu' if ds == 'FBDB15K/norm' else 'desalign_cpu'
    base = folder + '__' + ds.replace('/', '_') + '_block0_initial'
    emb = [np.load(R / 'frozen' / (base + ('' if i == 0 else f'_depth{i}') + '_node_embeddings.npy'), mmap_mode='r') for i in range(3)]

    def score():
        S = emb[0][left] @ emb[0][right].T
        for F in emb[1:]:
            S += F[left] @ F[right].T
        return S / 3
    host_start = dict(cpu_percent=psutil.cpu_percent(interval=1), available_ram_bytes=psutil.virtual_memory().available, process_count=len(psutil.pids()))
    S, scoretime = repeat(score)
    record = dict(dataset=ds, n=n, m=len(right), regime=regime, threads=2, hardware=dict(platform=platform.platform(), processor=platform.processor(), logical_cpus=os.cpu_count()), implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), host_load_start=host_start, score=scoretime, score_bytes=S.nbytes, embedding_array_bytes=sum((F.nbytes for F in emb)), decoders={}, protocol='protocol_systems.json', started_unix=time.time())
    dest.write_text(json.dumps(record, indent=2))
    print('SCORED ' + stem, flush=True)
    w = workload(ds, left, right, man['gallery'])
    pred = {}
    for name, fn in [('row', lambda: S.argmax(1)), ('dense', lambda: assignment(S)), ('stable', lambda: stable(S, verify=False)), ('sparse', lambda: sparse(S))]:
        output, timing = repeat(fn)
        if isinstance(output, tuple):
            p, ks = output
            timing['sparse_k_attempts'] = ks
            timing['stored_candidate_edges'] = n * ks[-1]
        else:
            p = output
        if name == 'stable':
            assert np.array_equal(p, stable(S))
            timing['expensive_validation_outside_timing'] = True
        pred[name] = p
        if name in ['sparse', 'stable']:
            timing['raw_objective_regret'] = float(np.sum(S[np.arange(n), pred['dense']].astype(float) - S[np.arange(n), p].astype(float)))
            assert timing['raw_objective_regret'] >= -0.001
        timing['query_metrics'] = {q: metrics(p, U, V, gold, q != 'cross_filter') for q, (U, V, _) in w.items()}
        record['decoders'][name] = timing
        dest.write_text(json.dumps(record, indent=2))
        print('TIMED ' + stem + ' ' + name + ' ' + str(timing['median_s']), flush=True)
    if regime == 'closed':
        P, timing = repeat(lambda: ot15(S))
        record['ot15'] = timing
        record['ot15']['marginal_residual'] = float(max(abs(P.sum(0) - 1).max(), abs(P.sum(1) - 1).max()))
        del P
        gc.collect()
        dest.write_text(json.dumps(record, indent=2))
    db = connect(ds)
    mapping = [(int(a), int(right[j])) for a, j in zip(left, pred['dense'])]

    def materialize():
        db.execute('DELETE FROM mapping')
        db.executemany('INSERT INTO mapping VALUES(?,?)', mapping)
    _, record['mapping_materialization'] = repeat(materialize)
    queries = w['coarse'][2]
    record['sqlite'] = dict(version=sqlite3.sqlite_version, existing_fact_indexes=True, query_plan=db.execute('EXPLAIN QUERY PLAN ' + sql('coarse', queries[0])[0], sql('coarse', queries[0])[1]).fetchall(), main_database_bytes=(DATA / 'sql' / (ds.replace('/', '_') + '.sqlite')).stat().st_size)

    def queries_sql():
        return [db.execute(*sql('coarse', key)).fetchall() for key in queries]
    answers_sql, record['sql_queries'] = repeat(queries_sql)
    record['sql_queries']['query_count'] = len(queries)
    record['sql_queries']['returned_rows'] = sum(map(len, answers_sql))
    record['sql_queries']['query_throughput_per_s'] = len(queries) / record['sql_queries']['median_s']
    db.close()
    record['complete'] = True
    record['finished_unix'] = time.time()
    record['peak_process'] = memory()
    record['host_load_end'] = dict(cpu_percent=psutil.cpu_percent(interval=1), available_ram_bytes=psutil.virtual_memory().available, process_count=len(psutil.pids()))
    dest.write_text(json.dumps(record, indent=2))
    np.savez_compressed(O / (stem + '_predictions.npz'), **pred)
    print('COMPLETE ' + stem, flush=True)
if __name__ == '__main__':
    main()
