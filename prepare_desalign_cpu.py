from pathlib import Path
import hashlib, json
R = Path(__file__).resolve().parent
src = R / 'thirdparty/DESAlign'
dst = R / 'desalign_cpu'
log = []
for folder in ['model', 'src']:
    for p in (src / folder).glob('*.py'):
        text = p.read_text(encoding='utf8')
        new = text.replace('.cuda()', '.cpu()')
        if p.name == 'layers.py':
            old = '            grad_a_dense = grad_output.matmul(b.t())\n            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]\n            grad_values = grad_a_dense.view(-1)[edge_idx]'
            repl = '            edges = a._indices()\n            grad_values = torch.empty(edges.shape[1], dtype=b.dtype, device=b.device)\n            for start in range(0, edges.shape[1], 4096):\n                r, c = edges[:, start:start+4096]\n                grad_values[start:start+4096] = (grad_output[r] * b[c]).sum(dim=1)'
            assert old in new
            new = new.replace(old, repl)
            old = '            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1)  # edge: 2*D x E\n            edge_e = torch.exp(-self.leaky_relu(edge_h.mm(self.a_src_dst[i]).squeeze()))  # edge_e: 1 x E'
            repl = '            source_score = h @ self.a_src_dst[i, :self.f_out]\n            target_score = h @ self.a_src_dst[i, self.f_out:]\n            edge_logits = source_score[edge[0]] + target_score[edge[1]]\n            edge_e = torch.exp(-self.leaky_relu(edge_logits.squeeze()))'
            assert old in new
            new = new.replace(old, repl)
        target = dst / folder / p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new, encoding='utf8')
        log.append(dict(source=str(p.relative_to(src)), original_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), port_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), changed=text != new))
(dst / 'port_manifest.json').write_text(json.dumps(dict(upstream_commit='d22eb74130c77d5c6bb29b1e47e5ee7286400155', changes=['CUDA tensor placement changed to CPU', 'Equivalent edge-only sparse-matmul gradient instead of dense NxN intermediate', 'Equivalent separable attention logit evaluation instead of concatenated edge features'], files=log), indent=2))
print('Prepared', len(log), 'source files')
