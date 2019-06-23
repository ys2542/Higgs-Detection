import logging
from sklearn.metrics import roc_auc_score, roc_curve

class Evaluate_Model(object):
  def __init__(self, y_true, y_pred, efficiency_value=0.5, name='Model'):
    self.y_true = y_true
    self.y_pred = y_pred
    self.p = efficiency_value
    self.name = name
    self.best_auc = 0.0
    self.best_fpr = 1.0

  def _get_efficiency(self):
    for i, tpr in enumerate(self.tpr):
      if tpr >= self.p:
        return self.fpr[i]
    raise Exception("No suitable FPR found")

  def evaluate(self):
    self.auc = roc_auc_score(self.y_true, self.y_pred)
    self.fpr, self.tpr, thresholds = roc_curve(self.y_true, self.y_pred)
    self.efficiency = self._get_efficiency()
    if self.auc > self.best_auc:
      self.best_auc = self.auc
      self.best_fpr = self.efficiency

  def log_scores(self):
    logging.warning(self.name+" scores:")
    logging.warning("  {:.4f} AUC".format(self.auc))
    logging.warning("  {:.3e} FPR at TPR = {}".format(self.efficiency, self.p))
    logging.warning("Best AUC: {:.4f}, Best FPR: {:.3e}".format(self.best_auc, self.best_fpr))

  def update(self, y_true, y_pred):
    self.y_true = y_true
    self.y_pred = y_pred
    
