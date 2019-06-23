from sklearn.linear_model import LogisticRegression

from evaluate.evaluate import Evaluate_Model

def train_and_evaluate(train_X, train_y, test_X, test_y):
  # Define model
  clf = LogisticRegression(
        tol=0.00001, 
        C=1000, 
        class_weight='balanced', 
        solver='sag', 
        max_iter=10000, 
        n_jobs=-1)
  # Train model
  clf.fit(train_X, train_y)
  # Predict on train, test set
  train_y_pred = clf.predict(train_X)
  test_y_pred  = clf.predict(test_X)
  # Set up evaluation for train, test
  train_eval = Evaluate_Model(train_y, train_y_pred, name="Logistic train")
  test_eval  = Evaluate_Model(test_y,  test_y_pred,  name="Logistic test")
  # Evaluate train, test
  train_eval.evaluate()
  test_eval.evaluate()
  # Log evaluation
  train_eval.log_scores()
  test_eval.log_scores()
