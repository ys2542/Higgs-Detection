import logging
from random import shuffle
import torch
import torch.nn.functional as f
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np

import models.mlps.gnn.model as model
from evaluate.evaluate import Evaluate_Model

class Model_Handler(object):
  def __init__(self, 
                nb_features, 
                nb_layers=2, 
                nb_hidden=20,
                lrate=0.001,
                lrate_decay=1.00,
                nb_epoch = 2,
                minibatch=10,
                ):
    self.net = model.GNN(nb_features, nb_layers, nb_hidden)
    self.lrate = lrate
    self.lrate_decay = lrate_decay
    self.nb_epoch = nb_epoch
    self.minibatch = minibatch
    print(self.net)
    self._set_cuda()

  def _set_cuda(self):
    if torch.cuda.is_available():
      self.cuda = True
      self.net.cuda()
      logging.warning("Working on GPU")
    else:
      self.cuda = False
      logging.warning("Working on CPU")

  def fit(self, train_X, train_y, test_X, test_y):
    nb_train = train_X.shape[0]
    nb_test  = test_X.shape[0]
    eval_train = Evaluate_Model(None, None, name='train')
    eval_test  = Evaluate_Model(None, None, name='test')
    criterion = f.binary_cross_entropy
    optimizer = Adam(self.net.parameters(), lr=self.lrate)
    logging.warning("Training on {} samples".format(nb_train))
    for i in range(self.nb_epoch):
      logging.info("\nEpoch {} lrate: {}".format(i+1, self.lrate))
      epoch_loss = self._train_one_epoch(train_X, train_y, criterion, optimizer)
      logging.info("Epoch loss: {0:.4e}".format(epoch_loss))

      # Evaluate Performance
      eval_train.update(train_y[:nb_test], self.pred(train_X[:nb_test]))
      eval_train.evaluate()
      eval_train.log_scores()
      eval_test.update(test_y, self.pred(test_X))
      eval_test.evaluate()
      eval_test.log_scores()

      self.lrate *= self.lrate_decay

  def pred(self, X_in):
    self.net.eval()
    nb_eval = X_in.shape[0]
    nb_batch = nb_eval // self.minibatch
    if (nb_eval / self.minibatch) != nb_batch:
      nb_batch += 1
    out = np.zeros((nb_eval,1))
    for i in range(nb_batch):
      idx_st = i*self.minibatch
      idx_fn = min((i+1)*self.minibatch, nb_eval)
      batch_idx = [j for j in range(idx_st, idx_fn)]
      batch_X = Variable(torch.FloatTensor(X_in[batch_idx]))
      batch_X = batch_X.resize(batch_X.size()[0],4,4)
      if self.cuda:
        batch_X = batch_X.cuda()
      batch_out = self.net(batch_X).data.cpu().numpy()
      out[batch_idx] = batch_out
    return out

  def _train_one_epoch(self, X, y, criterion, optimizer):
    self.net.train()
    idx = [i for i in range(X.shape[0])]
    shuffle(idx)

    nb_batch = X.shape[0] // self.minibatch
    nb_print = max(1, nb_batch // 10)

    step_loss = 0
    epoch_loss = 0
    for i in range(nb_batch):
      optimizer.zero_grad()
      idx_st = i*self.minibatch
      idx_fn = idx_st + self.minibatch
      batch_idx = idx[idx_st:idx_fn]
      batch_X = Variable(torch.Tensor(X[batch_idx])).resize(self.minibatch,4,4)
      batch_y = Variable(torch.Tensor([int(y[s:s+1]) for s in batch_idx]))

      if self.cuda:
        batch_X = batch_X.cuda()
        batch_y = batch_y.cuda()

      y_pred = self.net(batch_X)
      loss = criterion(y_pred,batch_y.resize(y_pred.size()[0],1))
      loss.backward()
      optimizer.step()

      step_loss  += loss.data[0]
      epoch_loss += loss.data[0]

      if ((i+1) % nb_print) == 0:
        logging.info("  {:d}: {:.4e}".format((i+1)*self.minibatch, step_loss/nb_print))
        step_loss = 0
    return epoch_loss / nb_batch
