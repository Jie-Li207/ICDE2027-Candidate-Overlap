import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from cdmea_cpu.models import RRGAT, scatter_sum

class CheckpointRRGAT(RRGAT):

    def forward(self, inputs):
        features, rel_emb, adj, r_index, r_val = inputs
        features = self.activation(features)
        outputs = [features]

        def layer(features, rel_emb, kernel):
            tri_rel = torch.sparse_coo_tensor(r_index, r_val, size=[self.triple_size, self.rel_size], dtype=features.dtype)
            tri_rel = torch.sparse.mm(tri_rel, rel_emb)
            tri_rel = F.normalize(tri_rel, dim=1, p=2)
            neighs = features[adj[1].long()]
            neighs = neighs - 2 * torch.sum(neighs * tri_rel, dim=1, keepdim=True) * tri_rel
            attention = torch.sparse_coo_tensor(adj, torch.squeeze(tri_rel @ kernel, -1), size=[self.node_size, self.node_size])
            attention = torch.sparse.softmax(attention, dim=1)
            new = scatter_sum(src=neighs * attention.coalesce().values().unsqueeze(-1), dim=0, index=adj[0].long())
            return self.activation(new)
        for kernel in self.attn_kernels:
            features = checkpoint(layer, features, rel_emb, kernel, use_reentrant=False) if self.training else layer(features, rel_emb, kernel)
            outputs.append(features)
        return torch.cat(outputs, dim=-1)
