import torch
import math
import torch.nn as nn

class Sin(nn.Module):
    def __init__(self, w0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
      return torch.sin(self.w0 * x)
    
def sine_init(layer, w0, is_first=False):
    with torch.no_grad():
        in_dim = layer.weight.size(-1)
        if is_first:
            bound = 1 / in_dim
        else:
            bound = math.sqrt(6 / in_dim) / w0
        layer.weight.uniform_(-bound, bound)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound)

class SynthesisNet(nn.Module):
    def __init__(self, features, w0_initial=30.0, w0=1.0):
        super().__init__()

        self.n_layers = len(features)-1
        w = lambda n: w0_initial if n == 0 else w0
        self.layers = nn.ModuleList([nn.Linear(features[n], features[n+1]) for n in range(self.n_layers)])
        self.activations = nn.ModuleList([Sin(w(n)) if n != (self.n_layers - 1) else None for n in range(self.n_layers)])

        for i, layer in enumerate(self.layers):
          if i == 0:
            sine_init(layer, w0_initial, is_first=True)
          else:
            sine_init(layer, w0)

    def forward(self, x, modulation):
      for i, (layer, act) in enumerate(zip(self.layers, self.activations)):

        x = layer(x)

        if act is not None:
           x = act(x)
           gamma = modulation[i][..., : x.shape[1]]
           beta = modulation[i][..., x.shape[1] :]
           x = x * gamma + beta
      
      return x

class ModulationNet(nn.Module):
  def __init__(self, features, z_emb_size):
    super().__init__()

    self.n_layers = len(features)-1
    self.z_emb_size = z_emb_size
    self.layers = nn.ModuleList([nn.Linear(features[n]+self.z_emb_size, features[n+1]) if n != 0 else nn.Linear(self.z_emb_size, features[n+1]) for n in range(self.n_layers)])
    self.activations = nn.ModuleList([nn.ReLU(inplace=True) if n != (self.n_layers-1) else None for n in range(self.n_layers)])

  def forward(self, z):
    input = z
    layerwise_outputs = []
    for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
      if i > 0:
        input = torch.cat([x, z], dim=1)
      
      x = layer(input)

      if act is not None: 
         x = act(x)

      layerwise_outputs.append(x)

    return layerwise_outputs