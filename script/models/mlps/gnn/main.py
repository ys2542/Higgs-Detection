import logging

from evaluate.evaluate import Evaluate_Model
import models.mlps.gnn.experiment_handler as eh

def _get_jets(X):
  X = X[:,5:21]
  return X

def train_and_evaluate(train_X, train_y, test_X, test_y, emb_features=False):
  # Select only the four jets
  train_X = _get_jets(train_X)
  test_X  = _get_jets(test_X)
  kw_args = {}
  kw_args['nb_epoch'] = 2000
  kw_args['lrate'] = 0.001
  kw_args['minibatch'] = 2000
  kw_args['nb_hidden'] = 64
  kw_args['lrate_decay'] = 0.98
  kw_args['nb_layers'] = 6
  # Define model
  clf = eh.Model_Handler(nb_features=4,**kw_args)
  # Train model
  logging.warning("Training")
  clf.fit(train_X, train_y, test_X, test_y)
  # clf.fit(train_X, train_y)
  # Predict on train, test set
  logging.warning("Predicting")
  train_y_pred = clf.pred(train_X)
  test_y_pred  = clf.pred(test_X)


  # Set up evaluation for train, test
  train_eval = Evaluate_Model(train_y, train_y_pred, name="Logistic train")
  test_eval  = Evaluate_Model(test_y,  test_y_pred,  name="Logistic test")
  # Evaluate train, test
  train_eval.evaluate()
  test_eval.evaluate()
  # Log evaluation
  train_eval.log_scores()
  test_eval.log_scores()
