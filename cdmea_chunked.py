import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from cdmea_cpu.models import RRGAT

class ReflectionAggregate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, features, relation, attention, adj, n):
        ctx.save_for_backward(features, relation, attention, adj)
        out = features.new_zeros((n, features.shape[1]))
        for start in range(0, len(relation), 8192):
            sl = slice(start, start + 8192)
            x = features[adj[1, sl]]
            r = relation[sl]
            dot = (x * r).sum(1, keepdim=True)
            out.index_add_(0, adj[0, sl], attention[sl, None] * (x - 2 * dot * r))
        return out

    @staticmethod
    def backward(ctx, grad):
        features, relation, attention, adj = ctx.saved_tensors
        dx = torch.zeros_like(features)
        dr = torch.empty_like(relation)
        da = torch.empty_like(attention)
        for start in range(0, len(relation), 8192):
            sl = slice(start, start + 8192)
            x = features[adj[1, sl]]
            r = relation[sl]
            g = grad[adj[0, sl]]
            a = attention[sl, None]
            xr = (x * r).sum(1, keepdim=True)
            gr = (g * r).sum(1, keepdim=True)
            dx.index_add_(0, adj[1, sl], a * (g - 2 * gr * r))
            dr[sl] = -2 * a * (gr * x + xr * g)
            da[sl] = (g * (x - 2 * xr * r)).sum(1)
        return (dx, dr, da, None, None)

class ReflectedAttentionLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, rel, kernel, adj, ri, rv, n, e, nr):
        Z = torch.sparse.mm(torch.sparse_coo_tensor(ri, rv, (e, nr), device=x.device), rel)
        length = Z.norm(p=2, dim=1, keepdim=True)
        r = Z / length.clamp_min(1e-12)
        del Z
        att = torch.sparse.softmax(torch.sparse_coo_tensor(adj, (r @ kernel).squeeze(-1), (n, n)), dim=1).coalesce()
        assert torch.equal(adj, att.indices())
        a = att.values()
        out = ReflectionAggregate.apply(x, r, a, adj, n)
        ctx.save_for_backward(x, rel, kernel, adj, ri, rv, r, length, a)
        return out

    @staticmethod
    def backward(ctx, g):
        x, rel, kernel, adj, ri, rv, r, length, a = ctx.saved_tensors
        n = len(x)
        e = len(r)
        da = torch.empty_like(a)
        rowmean = a.new_zeros(n)
        dx = torch.zeros_like(x)
        drel = torch.zeros_like(rel)
        dk = torch.zeros_like(kernel)
        for start in range(0, e, 8192):
            sl = slice(start, start + 8192)
            xx = x[adj[1, sl]]
            rr = r[sl]
            gg = g[adj[0, sl]]
            xr = (xx * rr).sum(1, keepdim=True)
            da[sl] = (gg * (xx - 2 * xr * rr)).sum(1)
            rowmean.index_add_(0, adj[0, sl], a[sl] * da[sl])
        for start in range(0, e, 8192):
            end = min(e, start + 8192)
            sl = slice(start, end)
            xx = x[adj[1, sl]]
            rr = r[sl]
            gg = g[adj[0, sl]]
            aa = a[sl, None]
            xr = (xx * rr).sum(1, keepdim=True)
            gr = (gg * rr).sum(1, keepdim=True)
            dl = aa * (da[sl, None] - rowmean[adj[0, sl], None])
            dx.index_add_(0, adj[1, sl], aa * (gg - 2 * gr * rr))
            dr = -2 * aa * (gr * xx + xr * gg) + dl * kernel.T
            dk += rr.T @ dl
            dz = (dr - rr * (dr * rr).sum(1, keepdim=True) * (length[sl] >= 1e-12)) / length[sl].clamp_min(1e-12)
            bounds = torch.searchsorted(ri[0], torch.tensor([start, end], device=ri.device))
            lo, hi = map(int, bounds)
            drel.index_add_(0, ri[1, lo:hi], rv[lo:hi, None] * dz[ri[0, lo:hi] - start])
        return (dx, drel, dk, None, None, None, None, None, None)

class ChunkedRRGAT(RRGAT):

    def forward(self, inputs):
        features, rel_emb, adj, r_index, r_val = inputs
        features = self.activation(features)
        outputs = [features]

        def layer(features, rel_emb, kernel):
            new = ReflectedAttentionLayer.apply(features, rel_emb, kernel, adj, r_index, r_val, self.node_size, self.triple_size, self.rel_size)
            return self.activation(new)
        for kernel in self.attn_kernels:
            features = checkpoint(layer, features, rel_emb, kernel, use_reentrant=False) if self.training else layer(features, rel_emb, kernel)
            outputs.append(features)
        return torch.cat(outputs, dim=-1)

def check():
    torch.manual_seed(23027)
    records = []
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        for n, d, e in [(17, 7, 101), (31, 13, 901), (37, 3, 18001)]:
            adj = torch.randint(n, (2, e), device=device)
            x = torch.randn(n, d, dtype=torch.float64, device=device, requires_grad=True)
            r = torch.randn(e, d, dtype=torch.float64, device=device, requires_grad=True)
            a = torch.rand(e, dtype=torch.float64, device=device, requires_grad=True)
            g = torch.randn(n, d, dtype=torch.float64, device=device)
            xx = x[adj[1]]
            rr = xx - 2 * (xx * r).sum(1, keepdim=True) * r
            ref = x.new_zeros((n, d)).index_add(0, adj[0], a[:, None] * rr)
            gr = torch.autograd.grad((ref * g).sum(), (x, r, a))
            out = ReflectionAggregate.apply(x, r, a, adj, n)
            got = torch.autograd.grad((out * g).sum(), (x, r, a))
            errors = [float((out - ref).abs().max())] + [float((aa - bb).abs().max()) for aa, bb in zip(got, gr)]
            assert max(errors) < 1e-09
            records.append(dict(device=device, n=n, d=d, edges=e, max_errors=errors))
    return records
if __name__ == '__main__':
    import json
    from pathlib import Path
    checks = check()
    (Path(__file__).parent / 'cdmea_v23_gpu/chunked_gradient_checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
