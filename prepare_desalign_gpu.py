from pathlib import Path
R = Path(__file__).resolve().parent
s = (R / 'prepare_desalign_cpu.py').read_text()
assert "dst = R / 'desalign_cpu'" in s
s = s.replace("dst = R / 'desalign_cpu'", "dst = R / 'desalign_gpu'").replace("new = text.replace('.cuda()', '.cpu()')", 'new = text').replace('CUDA tensor placement changed to CPU', 'Original CUDA placements retained')
exec(compile(s, str(R / 'prepare_desalign_gpu.py'), 'exec'))
