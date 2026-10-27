from pathlib import Path
import requests, json, bs4
R = Path(__file__).resolve().parent
out = R / 'public_data/desalign'
out.mkdir(exist_ok=True)
s = requests.Session()
u = 'https://drive.google.com/uc?id=1xYgXv8n1F8CBSNmYs5f3GfdJmrWHpxDJ&export=download'
r = s.get(u, timeout=25)
r.raise_for_status()
soup = bs4.BeautifulSoup(r.text, 'html.parser')
f = soup.find('form')
assert f
args = {e['name']: e.get('value', '') for e in f.select('input[name]')}
url = f['action']
rr = s.get(url, params=args, headers={'Range': 'bytes=-65536'}, stream=True, timeout=40)
meta = dict(status=rr.status_code, headers=dict(rr.headers), source=u, download_url=rr.url)
print(json.dumps({k: v for k, v in meta.items() if k != 'download_url'}, indent=2))
if rr.status_code == 206:
    data = rr.content
    (out / 'tail.bin').write_bytes(data)
    meta['bytes'] = len(data)
rr.close()
(out / 'range_probe.json').write_text(json.dumps(meta, indent=2))
