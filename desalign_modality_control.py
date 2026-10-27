from desalign_baseline_v9 import DESAlign

class NoImageDESAlign(DESAlign):

    def inner_view_loss(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill, vir_emb=None, weight_norm=None):
        if weight_norm is None or self.args.w_img:
            return super().inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill, vir_emb, weight_norm)
        assert img_emb is None and vir_emb is None and (name_emb is None) and (char_emb is None)
        assert weight_norm.shape[1] == 3
        w = weight_norm * 3
        losses = [self.criterion_cl(gph_emb, train_ill, weight_norm=w[:, 2]), self.criterion_cl(rel_emb, train_ill, weight_norm=w[:, 1]), self.criterion_cl(att_emb, train_ill, weight_norm=w[:, 0]), 0, 0, 0]
        return self.multi_loss_layer(losses)
