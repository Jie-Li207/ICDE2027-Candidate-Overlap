import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import requests
R = Path(__file__).resolve().parent

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(x)
    return h.hexdigest()

def fetch(url, p, digest):
    if p.exists():
        assert sha(p) == digest, (p, 'existing hash mismatch')
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.partial')
    with requests.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        with tmp.open('wb') as out:
            for chunk in r.iter_content(4 * 1024 * 1024):
                out.write(chunk)
    assert sha(tmp) == digest, (p, 'download hash mismatch')
    tmp.replace(p)

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--verify-only', action='store_true')
    args = a.parse_args()
    checked = []
    for name, url, commit in [('DESAlign', 'https://github.com/wyy-code/DESAlign.git', 'd22eb74130c77d5c6bb29b1e47e5ee7286400155'), ('CDMEA', 'https://github.com/sutaoyu/CDMEA.git', 'c9929cc143d32ea6b9af97069c6f1d43c206c08d')]:
        p = R / 'thirdparty' / name
        if not p.exists() and (not args.verify_only):
            p.parent.mkdir(exist_ok=True)
            subprocess.run(['git', 'clone', url, str(p)], check=True)
            subprocess.run(['git', '-C', str(p), 'checkout', '--detach', commit], check=True)
        assert p.exists(), str(p)
        assert subprocess.check_output(['git', '-C', str(p), 'rev-parse', 'HEAD'], text=True).strip() == commit
        checked.append(dict(repository=name, commit=commit))
    d = R / 'public_data/mmkb'
    for f in ['downloads.json', 'metadata_downloads.json']:
        for row in json.loads((d / f).read_text()):
            name = row['name']
            name = name + '.h5' if name.endswith('_visual') else 'sameAs.txt' if name == 'sameAs' else name
            p = d / name
            if not args.verify_only:
                fetch(row['url'], p, row['sha256'])
            assert p.exists() and sha(p) == row['sha256'], str(p)
            checked.append(dict(file=str(p.relative_to(R)), sha256=row['sha256']))
    manifest = json.loads((R / 'public_data/desalign/verified_downloads.json').read_text())
    missing = [row for row in manifest['members'] if not (R / 'public_data/desalign/selected' / row['name']).exists()]
    if missing and (not args.verify_only):
        subprocess.run([sys.executable, 'probe_desalign_data.py'], cwd=R, check=True)
        from remote_desalign_zip import Remote
        import zipfile
        with zipfile.ZipFile(Remote()) as z:
            for row in missing:
                p = (R / 'public_data/desalign/selected' / row['name']).resolve()
                assert p.is_relative_to((R / 'public_data/desalign/selected').resolve())
                data = z.read(row['name'])
                assert hashlib.sha256(data).hexdigest() == row['sha256']
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
    for row in manifest['members']:
        p = R / 'public_data/desalign/selected' / row['name']
        assert p.exists() and sha(p) == row['sha256'], str(p)
        checked.append(dict(file=str(p.relative_to(R)), sha256=row['sha256']))
    if not args.verify_only:
        for script in ['prepare_desalign_cpu.py', 'prepare_desalign_gpu.py', 'prepare_cdmea_cpu.py', 'entity_externality.py']:
            subprocess.run([sys.executable, script], cwd=R, check=True)
    (R / 'bootstrap_verification.json').write_text(json.dumps(dict(checked=checked, verify_only=args.verify_only), indent=2))
    print('VERIFIED', len(checked), 'inputs')
if __name__ == '__main__':
    main()
