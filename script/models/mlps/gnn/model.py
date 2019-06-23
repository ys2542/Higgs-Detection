import torch
import torch.nn as nn
import torch.nn.functional as f

class GNN(nn.Module):
  def __init__(self, nb_features, nb_layers=2, nb_hidden=32):
    super(GNN, self).__init__()
    self.first_layer = ResGNN(nb_features, nb_hidden)
    layers = [ResGNN(nb_hidden, nb_hidden) for _ in range(nb_layers-1)]
    self.layers = nn.ModuleList(layers)
    self.kernel = GaussianSoftmax(nb_features)
    self.readout = Sum(nb_hidden)

  def forward(self, X):
    adj = self.kernel(X)
    emb = self.first_layer(X, adj)
    for layer in self.layers:
      emb = layer(emb, adj)
    emb = self.readout(emb)
    return emb

#########
# KERNELS
#########

def _softmax(A):
  A = torch.exp(A)
  S = A.sum(2,keepdim=True)
  return A / S.expand_as(A)

class GaussianSoftmax(nn.Module):
  def __init__(self, *args, **kwargs):
    super(GaussianSoftmax, self).__init__()
    self.sigma = nn.Parameter(torch.rand(1)+1.0)

  def forward(self, X):
    batch, nb_node, fmap = X.size()
    coord = X.unsqueeze(1).repeat(1, nb_node, 1, 1)
    sqdist = (coord-coord.transpose(1,2))**2
    sqdist = sqdist.sum(3)
    gaussian = torch.exp(-sqdist / self.sigma)
    return _softmax(gaussian)

class MLPSigmoid(nn.Module):
  def __init__(self, fmap, nb_hidden=32):
    super(MLPSigmoid, self).__init__()
    self.fc1 = nn.Linear(2*fmap, nb_hidden)
    self.fc2 = nn.Linear(nb_hidden, 1)
    self.act = nn.Tanh()

  def forward(self, X):
    batch, nb_node, fmap = X.size()
    # Cartesian product of X
    r1 = X.repeat(1,nb_node,1)
    r2 = X.repeat(1,1,nb_node).resize(batch,nb_node*nb_node,fmap)
    adj = torch.cat((r1,r2),2)
    adj = self.act(self.fc1(adj))
    adj = self.fc2(adj)
    adj = adj.resize(batch,nb_node,nb_node)
    return f.sigmoid(adj)

###################
# GRAPH CONVOLUTION
###################

def op_avg(X, A):
  return X.sum(1,keepdim=True).expand_as(X)

def op_degree(X, A):
  batch, nb_node, fmap = X.size()
  I = torch.autograd.Variable(torch.eye(nb_node)).unsqueeze(0).repeat(batch,1,1)
  if X.is_cuda:
    I = I.cuda()
  D = A.sum(2,keepdim=True).expand_as(A) * I
  return torch.matmul(D, X)

def op_identity(X, A):
  return X

def op_adj(X, A):
  return torch.matmul(A, X)
  

class ResGNN(nn.Module):
  def __init__(self, fmap_in, fmap_out):
    super(ResGNN, self).__init__()
    # self.ops = [op_avg, op_degree, op_identity, op_adj]
    self.ops = [op_identity, op_adj]
    self.nb_op = len(self.ops)
    self.lin  = Simple(fmap_in, fmap_out//2, nb_op=self.nb_op, ops=self.ops)
    self.nlin = Simple(fmap_in, fmap_out//2, self.nb_op, f.relu, ops=self.ops)

  def forward(self, X, A):
    batch, nb_node, fmap = X.size()

    lin  = self.lin(X, A)
    nlin = self.nlin(X, A)
    return torch.cat((lin, nlin), 2)


class Simple(nn.Module):
  def __init__(self, fmap_in, fmap_out, nb_op=1, activation=None, ops=[op_adj]):
    super(Simple, self).__init__()
    self.fc = nn.Linear(nb_op * fmap_in, fmap_out)
    self.activation=activation
    self.ops=ops

  def forward(self, X, A):
    batch, nb_node, fmap_in = X.size()
    emb = tuple(op(X,A) for op in self.ops)
    emb = torch.cat(emb,2)
    emb = self.fc(emb)
    if self.activation is not None:
      emb = self.activation(emb)
    return emb

#########
# Readout
#########

class Readout(nn.Module):
  def __init__(self, fmap):
    super(Readout, self).__init__()
    self.fc = nn.Linear(fmap, 1)
    self.act = nn.Sigmoid()

  def _pooling(self, X):
    raise Exception("Must use child readout class")

  def forward(self, X):
    emb = self._pooling(X)
    emb = self.act(self.fc(emb))
    return emb

class Sum(Readout):
  def __init(self, fmap):
    super(Sum, self).__init__(fmap)

  def _pooling(self, X):
    return X.sum(1)
