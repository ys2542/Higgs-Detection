import os
import csv
import logging
import numpy as np

import utils 

def _get_train_test_indices(nb_train, nb_test, nb_samples):
  train_idx = np.arange(nb_samples-nb_test)
  train_idx = np.random.permutation(train_idx)[:nb_train]

  test_idx = np.arange(nb_samples)[-nb_test:]
  return train_idx, test_idx

def _get_raw_samples_from_file(datafile, indices):
  indices.sort()
  count = 0
  samples = []
  with open(datafile, newline='') as csvfile:
    dataloader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(dataloader):
      if i == indices[count]:
        samples.append(list(map(float,row)))
        count += 1
        if count == len(indices):
          return samples
      if (i % 200000) == 0:
        logging.info("{} of {}".format(i, indices[-1]))

def _process_samples(raw_samples):
  nb_samples = len(raw_samples)
  nb_features = len(raw_samples[0])-1
  y = np.zeros(shape=nb_samples, dtype=int)
  X = np.zeros(shape=(nb_samples, nb_features))
  for i, sample in enumerate(raw_samples):
    y[i] = int(sample[0])
    X[i] = sample[1:]
  return X, y

def generate_samples(nb_samples_in_file, nb_train, nb_test, loadfile, savepath):
  if (nb_train+nb_test) > nb_samples_in_file:
    logging.error("Too many train, test samples requested for dataset size")
    exit()
  logging.warning("Generating {} train samples, {} test samples".format(nb_train, nb_test))
  train_idx, test_idx = _get_train_test_indices(nb_train, nb_test, nb_samples_in_file)
  train_raw_samples = _get_raw_samples_from_file(loadfile,train_idx)
  test_raw_samples  = _get_raw_samples_from_file(loadfile,test_idx)
  train_X, train_y = _process_samples(train_raw_samples)
  test_X,  test_y  = _process_samples(test_raw_samples)

  utils.save_data((train_X, train_y), os.path.join(savepath, 'train.pickle'))
  utils.save_data((test_X,  test_y),  os.path.join(savepath, 'test.pickle'))

if __name__ == "__main__":
  nb_samples_in_file = 10
  nb_train = 6
  nb_test = 4
  filename = 'HIGGS.csv'
