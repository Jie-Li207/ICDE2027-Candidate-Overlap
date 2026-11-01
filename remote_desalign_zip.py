from pathlib import Path
import io, json, requests, zipfile, hashlib, sys
R = Path(__file__).resolve().parent
O = R / 'public_data/desalign'

class Remote(io.RawIOBase):

    def __init__(self):
        meta = json.loads((O / 'range_probe.json').read_text())
        self.url = meta['download_url']
        self.total = int(meta['headers']['Content-Range'].split('/')[-1])
        self.pos = 0
        self.session = requests.Session()
        self.downloaded = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.total + offset
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.total - self.pos
        n = min(n, self.total - self.pos)
        if n <= 0:
            return b''
        start = self.pos
        end = start + n - 1
        rr = self.session.get(self.url, headers={'Range': f'bytes={start}-{end}'}, timeout=90)
        rr.raise_for_status()
        assert rr.status_code == 206 and rr.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/')
        data = rr.content
        assert len(data) == n
        self.downloaded += n
        self.pos += n
        return data

def main():
    f = Remote()
    z = zipfile.ZipFile(f)
    entries = [dict(name=i.filename, bytes=i.file_size, compressed=i.compress_size, crc=i.CRC) for i in z.infolist()]
    (O / 'archive_inventory.json').write_text(json.dumps(entries, indent=2))
    if len(sys.argv) == 1:
        for e in entries:
            if 'FB' in e['name'] or e['name'].endswith('training_attrs_1'):
                print(json.dumps(e))
    else:
        selected = [e for e in entries if any((s in e['name'] for s in sys.argv[1:])) and (not e['name'].endswith('/'))]
        log = []
        for e in selected:
            dest = (O / 'selected' / e['name']).resolve()
            assert dest.is_relative_to((O / 'selected').resolve())
            if dest.exists() and dest.stat().st_size == e['bytes']:
                print('EXISTS ' + e['name'], flush=True)
                continue
            print('FETCH ' + json.dumps(e), flush=True)
            data = z.read(e['name'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            log.append(dict(**e, sha256=hashlib.sha256(data).hexdigest()))
            (O / 'selected_downloads.json').write_text(json.dumps(log, indent=2))
            print('SAVED ' + e['name'], flush=True)
    print('HTTP_BYTES ' + str(f.downloaded))
if __name__ == '__main__':
    main()
