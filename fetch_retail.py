import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
import pandas as pd
import requests
R = Path(__file__).resolve().parent
O = R / 'public_data' / 'retail'
O.mkdir(parents=True, exist_ok=True)
REPO = 'blamm/retail_visual_rag_pipeline'
REVISION = 'bfb2f6e411b2e0976c899bb5618eb629f94c7b04'

def get(url):
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=45)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)

def main():
    f = O / 'source.json'
    if not f.exists():
        f.write_text(json.dumps(dict(repo=REPO, revision=REVISION, accessed='2026-10-31', license='cc-by-nc-nd-4.0', paper='https://openaccess.thecvf.com/content/CVPR2025W/FGVC/papers/Lamm_A_Visual_RAG_Pipeline_for_Few-Shot_Fine-Grained_Product_Classification_CVPRW_2025_paper.pdf'), indent=2))
    revision = json.loads(f.read_text())['revision']
    assert revision == REVISION
    base = 'https://huggingface.co/datasets/' + REPO + '/resolve/' + revision + '/'
    meta = O / 'test.parquet'
    if not meta.exists():
        meta.write_bytes(get(base + 'test.parquet').content)
    df = pd.read_parquet(meta)
    assert hashlib.sha256(meta.read_bytes()).hexdigest() == 'a042b72d2f5c8fce889d20a4df25422078474f89d5c5b70b7ae4140237144c4a'
    assert len(df) == 1101 and len(df['class'].unique()) == 367

    def one(row):
        rel = 'test/' + str(row['class']) + '/' + row['filename']
        p = O / rel
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            data = get(base + rel).content
            p.write_bytes(data)
        data = p.read_bytes()
        return dict(path=rel, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for entry in pool.map(one, df.to_dict('records')):
            records.append(entry)
            if len(records) % 100 == 0:
                print('DOWNLOADED', len(records), sum((x['bytes'] for x in records)), flush=True)
    assert hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest() == '7fac8c5e6cc9786daf2291b36ef586a3fd9dc12494ceba2b1862fd846126f07c'
    (O / 'files.json').write_text(json.dumps(dict(metadata_sha256=hashlib.sha256(meta.read_bytes()).hexdigest(), images=records), indent=2))
    print('COMPLETE', len(records), sum((x['bytes'] for x in records)), flush=True)
if __name__ == '__main__':
    main()
