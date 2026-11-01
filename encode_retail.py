import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
R = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(R / 'public_data' / 'model_cache'))
sys.path.insert(0, str(R / 'deps_retail'))
import open_clip
D = R / 'public_data' / 'retail'
O = R / 'query_scope'
CHECKPOINT = Path(os.environ.get('ICDE_CLIP_CHECKPOINT', str(R / 'public_data' / 'models' / 'open_clip_model.safetensors')))
MODEL_URL = 'https://huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/a6f597a30f7b82c51704746581f9a4e41421e878/open_clip_model.safetensors'

def main():
    O.mkdir(exist_ok=True)
    assert (D / 'files.json').exists()
    if (O / 'retail_features.npz').exists():
        print('EXISTS', flush=True)
        return
    if not CHECKPOINT.exists():
        import requests
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(MODEL_URL, stream=True, timeout=90)
        response.raise_for_status()
        with CHECKPOINT.open('wb') as f:
            for chunk in response.iter_content(4 * 1024 * 1024):
                f.write(chunk)
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == 'e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31'
    while True:
        status = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True)
        util, mem = map(int, status.splitlines()[0].split(','))
        if util < 15 and mem < 1000:
            break
        print('WAIT GPU', util, mem, flush=True)
        time.sleep(15)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _, transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained=str(CHECKPOINT), force_quick_gelu=True, device='cuda', precision='fp32')
    model.eval()
    files = json.loads((D / 'files.json').read_text())['images']
    pixels = []
    features = []
    t = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(files), 32):
            batch = []
            for item in files[start:start + 32]:
                p = D / item['path']
                assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256']
                with Image.open(p) as im:
                    im = im.convert('RGB')
                    pixels.append(hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest())
                    batch.append(transform(im))
            x = torch.stack(batch).cuda()
            y = model.encode_image(x, normalize=True).cpu().numpy()
            assert np.isfinite(y).all()
            features.append(y)
            if start % 256 == 0:
                print('ENCODED', start + len(batch), flush=True)
    np.savez_compressed(O / 'retail_features.npz', features=np.concatenate(features), paths=np.array([f['path'] for f in files]), pixel_sha256=np.array(pixels))
    record = dict(model='CLIP ViT-B/32 OpenAI', open_clip_version=open_clip.__version__, checkpoint=str(CHECKPOINT), checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), public_checkpoint='https://huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/a6f597a30f7b82c51704746581f9a4e41421e878/open_clip_model.safetensors', force_quick_gelu=True, precision='fp32', image_count=len(files), seconds=time.perf_counter() - t, peak_gpu_bytes=torch.cuda.max_memory_allocated(), data_source=json.loads((D / 'source.json').read_text()), feature_file_sha256=hashlib.sha256((O / 'retail_features.npz').read_bytes()).hexdigest(), pixel_hashes=pixels)
    (O / 'retail_feature_audit.json').write_text(json.dumps(record, indent=2))
    print('COMPLETE', len(files), record['seconds'], flush=True)
if __name__ == '__main__':
    main()
