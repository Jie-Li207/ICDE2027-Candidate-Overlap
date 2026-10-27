import requests, hashlib, json, zipfile, io
from pathlib import Path
R = Path(__file__).resolve().parent
O = R / 'public_data/products_v35'
O.mkdir(parents=True, exist_ok=True)
records = []
expected = {x['dataset']: x for x in json.loads((R / 'products_v35_expected.json').read_text())}
for name in ['Abt-Buy', 'Amazon-GoogleProducts']:
    url = f'https://dbs.uni-leipzig.de/files/datasets/{name}.zip'
    dest = O / (name + '.zip')
    if not dest.exists():
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        assert len(r.content) < 20000000
        dest.write_bytes(r.content)
    data = dest.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected[name]['sha256'], 'Upstream archive differs from recorded experiment: ' + name
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None
        members = []
        for item in z.infolist():
            if item.is_dir():
                continue
            assert not Path(item.filename).is_absolute() and '..' not in Path(item.filename).parts
            if item.filename.lower().endswith('.csv'):
                p = O / name / Path(item.filename).name
                p.parent.mkdir(exist_ok=True)
                raw = z.read(item)
                p.write_bytes(raw)
                members.append(dict(name=item.filename, file=p.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    records.append(dict(dataset=name, url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), members=members))
(O / 'provenance.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
