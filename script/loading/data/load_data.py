import os
import pickle
import logging
import numpy as np

import loading.data.preprocess as preproc

nb_samples_in_file = 11000000
nb_train_save = 4000000
nb_test_save  = 500000


def _load_file(filepath):
  with open(filepath,'rb') as filein:
    X, y = pickle.load(filein)
  return X, y

def _crop_to_requested_size(X, y, nb_samples, name):
  if y.shape[0] < nb_samples:
    logging.warning("Requested {} {} samples but only {} available".format(nb_samples, name, y.shape[0]))
    return X, y
  else:
    # Crop randomly
    idx = np.random.permutation(np.arange(y.shape[0]))
    idx = idx[:nb_samples]
    return X[idx], y[idx]
  

def load_data(savedir, raw_datafile, nb_train, nb_test):
  logging.info("Loading data...")
  trainfile = os.path.join(savedir, "train.pickle")
  testfile  = os.path.join(savedir, "test.pickle")
  try:
    train_X, train_y = _load_file(trainfile)
    test_X,  test_y  = _load_file(testfile)
  except:
    logging.warning("Train or test pickle files not found")
    preproc.generate_samples(nb_samples_in_file, nb_train_save, nb_test_save, raw_datafile, savedir)
    train_X, train_y = _load_file(trainfile)
    test_X,  test_y  = _load_file(testfile)
  # Crop to requested number of samples
  train_X, train_y = _crop_to_requested_size(train_X, train_y, nb_train, 'train')
  test_X,  test_y  = _crop_to_requested_size(test_X,  test_y,  nb_test,  'test')
  logging.info("Data loaded")
  return train_X, train_y, test_X, test_y

    
