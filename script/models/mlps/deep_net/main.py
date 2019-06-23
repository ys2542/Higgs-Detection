import logging

from evaluate.evaluate import Evaluate_Model
import models.mlps.deep_net.experiment_handler as eh

def train_and_evaluate(train_X, train_y, test_X, test_y):
  kw_args = {}
  kw_args['nb_epoch'] = 2000
  kw_args['lrate'] = 0.005
  kw_args['minibatch'] = 256
  kw_args['nb_hidden'] = 300
  kw_args['lrate_decay'] = 0.96
  kw_args['nb_layers'] = 5
  # Define model

  # Training on jets along
  # Comment to use all features
  # train_X = train_X[:,5:21]
  # test_X = test_X[:,5:21]

  logging.info(kw_args)
  clf = eh.Model_Handler(nb_features=train_X.shape[1],**kw_args)
  # Train model
  logging.warning("Training")
  clf.fit(train_X, train_y, test_X, test_y)
  # clf.fit(train_X, train_y)
  # Predict on train, test set
  logging.warning("Predicting")
  train_y_pred = clf.pred(train_X)
  test_y_pred  = clf.pred(test_X)


  # Set up evaluation for train, test
  train_eval = Evaluate_Model(train_y, train_y_pred, name="Deep NN")
  test_eval  = Evaluate_Model(test_y,  test_y_pred,  name="Deep NN")
  # Evaluate train, test
  train_eval.evaluate()
  test_eval.evaluate()
  # Log evaluation
  train_eval.log_scores()
  test_eval.log_scores()
