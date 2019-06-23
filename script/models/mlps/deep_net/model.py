import torch
import torch.nn as nn
import torch.nn.functional as f

def get_layers(nb_features, nb_layers, nb_hidden):
  layers = [nn.Linear(nb_features, nb_hidden)]
  for _ in range(nb_layers-1):
    layers.append(nn.Linear(nb_hidden, nb_hidden))
  layers.append(nn.Linear(nb_hidden, 1))
  return nn.ModuleList(layers)



class Deep_Net(nn.Module):
  def __init__(self, nb_features, nb_layers=2, nb_hidden=20):
    super(Deep_Net, self).__init__()
    self.activation = nn.ReLU()
    self.layers = get_layers(nb_features, nb_layers, nb_hidden)

  def forward(self, X):
    out = X
    for i in range(len(self.layers)-1):
      out = self.activation(self.layers[i](out))
    return f.sigmoid(self.layers[-1](out))


