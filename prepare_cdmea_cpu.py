from pathlib import Path
import json, hashlib, subprocess
R = Path(__file__).resolve().parent
S = R / 'thirdparty/CDMEA/src'
O = R / 'cdmea_cpu'
O.mkdir(exist_ok=True)
manifest = []
for name in ['models.py', 'utils.py', 'layers.py', 'loss.py', 'Load.py']:
    original = (S / name).read_text(encoding='utf8')
    text = original
    if name == 'models.py':
        text = text.replace('from torch_scatter import scatter_sum', 'def scatter_sum(src, index, dim=0):\n    assert dim == 0\n    out = src.new_zeros((int(index.max()) + 1,) + src.shape[1:])\n    return out.index_add(0, index, src)')
        start = text.index('try:\n    from utils import *')
        end = text.index('\n\ndef cosine_sim', start)
        text = text[:start] + 'from .utils import *\nfrom .layers import *\nfrom .loss import *\n' + text[end:]
    if name == 'loss.py':
        start = text.index('try:')
        end = text.index('\n\nclass InfoNCE_loss', start)
        text = text[:start] + 'import torch.nn.functional as F\n' + text[end:]
        text = text.replace('.cuda(self.device)', '.to(self.device)')
    (O / name).write_text(text, encoding='utf8')
    manifest.append(dict(file=name, original_sha256=hashlib.sha256(original.encode()).hexdigest(), cpu_sha256=hashlib.sha256(text.encode()).hexdigest()))
(O / '__init__.py').write_text('')
meta = dict(upstream_commit=subprocess.check_output(['git', '-C', str(R / 'thirdparty/CDMEA'), 'rev-parse', 'HEAD'], text=True).strip(), files=manifest, changes=['Native PyTorch index_add substitutes scatter_sum at dim0', 'Loss labels placed on configured CPU', 'Package-relative imports; circular unused model import removed from loss'])
(O / 'port_manifest.json').write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
