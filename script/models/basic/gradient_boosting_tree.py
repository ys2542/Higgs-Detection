'''
Created on Mar 28, 2018

@author: Yiyun Hu
'''
from sklearn.ensemble import GradientBoostingClassifier

from evaluate.evaluate import Evaluate_Model

def train_and_evaluate(train_X, train_y, test_X, test_y):
    # Define model
#     clf = GradientBoostingClassifier()
    clf = GradientBoostingClassifier(
        loss='exponential',
        learning_rate=0.01, 
        n_estimators=1000, 
        min_samples_split=30, 
        min_samples_leaf=10, 
        min_weight_fraction_leaf=0.01, 
        max_depth=5, 
        )

    # Train model
    clf.fit(train_X, train_y)
    # Predict on train, test set
    train_y_pred = clf.predict(train_X)
    test_y_pred  = clf.predict(test_X)
    # Set up evaluation for train, test
    train_eval = Evaluate_Model(train_y, train_y_pred, name="GBC train")
    test_eval  = Evaluate_Model(test_y,  test_y_pred,  name="GBC test")
    # Evaluate train, test
    train_eval.evaluate()
    test_eval.evaluate()
    # Log evaluation
    train_eval.log_scores()
    test_eval.log_scores()
