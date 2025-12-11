import torch.nn as nn
import torch

class INRWrapper(nn.Module):
  def __init__(self, SynthesisNet, ModulationNet, CNNEncoderBlock, total_train_samples, imp_latent_size, is_implicit, is_merge):
    super().__init__()

    self.synnet = SynthesisNet
    self.modnet = ModulationNet
    self.encoder = CNNEncoderBlock
    self.is_implicit = is_implicit
    self.is_merge = is_merge

    assert self.synnet.n_layers == self.modnet.n_layers + 1
    self.n_layers = self.synnet.n_layers

    if self.is_implicit or self.is_merge:
      self.implicit_embedding = nn.Embedding(total_train_samples, imp_latent_size)
      nn.init.normal_(self.implicit_embedding.weight, mean=0.0, std=0.01)

  def forward(self, field_t, x, indices):

    B = field_t.shape[0]
    n_pts = x.shape[0]

    if not self.is_implicit or self.is_merge:
      z_explicit = self.encoder(field_t)
      z_explicit_expand = (z_explicit.unsqueeze(1).repeat(1, n_pts, 1)).reshape(B * n_pts, -1)
      z_emb = z_explicit_expand

    if self.is_implicit or self.is_merge:
      z_implicit = self.implicit_embedding(indices)
      z_implicit_expand = (z_implicit.unsqueeze(1).repeat(1, n_pts, 1)).reshape(B * n_pts, -1)
      
      if self.is_merge:
        z_emb = torch.cat((z_explicit_expand, z_implicit_expand), dim=-1)
      else:
        z_emb = z_implicit_expand

    x_expand = x.unsqueeze(0).repeat(B, 1, 1).reshape(B * n_pts, -1)
    modulation = self.modnet(z_emb)
    out = self.synnet(x_expand, modulation)

    return out

  def forward_ito(self, field_t, x, z, full_embed = False):

    B = field_t.shape[0]
    n_pts = x.shape[0]

    if not full_embed:
      if not self.is_implicit or self.is_merge:
        z_explicit = self.encoder(field_t)
        z_explicit_expand = (z_explicit.unsqueeze(1).repeat(1, n_pts, 1)).reshape(B * n_pts, -1)
        z_emb = z_explicit_expand

      if self.is_implicit or self.is_merge:
        z_implicit_expand = (z.unsqueeze(1).repeat(1, n_pts, 1)).reshape(B * n_pts, -1)
        if self.is_merge:
          z_emb = torch.cat((z_explicit_expand, z_implicit_expand), dim=-1)
        else:
          z_emb = z_implicit_expand
      
    else:
      z_emb = z.unsqueeze(1).repeat(1, n_pts, 1).reshape(B * n_pts, -1)

    x_expand = x.unsqueeze(0).repeat(B, 1, 1).reshape(B * n_pts, -1)

    modulation = self.modnet(z_emb)
    out = self.synnet(x_expand, modulation)

    return out
     