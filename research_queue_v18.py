import json, subprocess, sys, time, ctypes
from pathlib import Path
R = Path(__file__).resolve().parent
O = R / 'queue_v18'
O.mkdir(exist_ok=True)

class Memory(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(k, ctypes.c_ulonglong) for k in ['total', 'available', 'total_page', 'available_page', 'total_virtual', 'available_virtual', 'extended']]

def available():
    m = Memory()
    m.length = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.available

def main():
    jobs = json.loads((R / 'protocol_v18_replication.json').read_text())['jobs']
    states = []
    for j, (dataset, block, noimage) in enumerate(jobs):
        stem = dataset.replace('/', '_') + f'_block{block}' + ('_noimage' if noimage else '')
        if (R / 'desalign_v11' / f'{stem}_development.json').exists():
            states.append(dict(job=j, state='already_complete', stem=stem))
            continue
        while available() < 7 * 1024 ** 3:
            time.sleep(30)
        cmd = [sys.executable, '-u', str(R / 'desalign_crosskg_v11.py'), '--dataset', dataset, '--block', str(block)] + (['--no-image'] if noimage else [])
        record = dict(job=j, stem=stem, state='running', started=time.time(), command=cmd)
        states.append(record)
        (O / 'status.json').write_text(json.dumps(states, indent=2))
        print('START ' + stem, flush=True)
        with open(O / f'job{j}.log', 'w', encoding='utf8') as f:
            result = subprocess.run(cmd, cwd=R, stdout=f, stderr=subprocess.STDOUT)
        record.update(state='complete' if result.returncode == 0 else 'failed', returncode=result.returncode, finished=time.time())
        (O / 'status.json').write_text(json.dumps(states, indent=2))
        print(record['state'].upper() + ' ' + stem, flush=True)
        if result.returncode:
            break
if __name__ == '__main__':
    main()
