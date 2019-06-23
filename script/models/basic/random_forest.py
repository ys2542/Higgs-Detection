import logging
from sklearn.ensemble import RandomForestClassifier

from evaluate.evaluate import Evaluate_Model

def train_and_evaluate(train_X, train_y, test_X, test_y):
  # Define model
  clf = RandomForestClassifier(
        n_estimators=50, 
        criterion='gini', 
        max_features=None, 
        max_depth=None, 
        min_samples_split=30, 
        min_samples_leaf=15, 
        min_weight_fraction_leaf=0.003,
        n_jobs=-1, 
        class_weight='balanced')
  # Train model
  logging.warning("Training")
  clf.fit(train_X, train_y)
  # Predict on train, test set
  logging.warning("Predicting")
  train_y_pred = clf.predict(train_X)
  test_y_pred  = clf.predict(test_X)
  # Set up evaluation for train, test
  train_eval = Evaluate_Model(train_y, train_y_pred, name="RandomForest train")
  test_eval  = Evaluate_Model(test_y,  test_y_pred,  name="RandomForest test")
  # Evaluate train, test
  train_eval.evaluate()
  test_eval.evaluate()
  # Log evaluation
  train_eval.log_scores()
  test_eval.log_scores()
