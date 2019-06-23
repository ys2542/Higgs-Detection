import argparse

def read_args():
  parser = argparse.ArgumentParser("Arguments for model and option selection")
  add_arg = parser.add_argument

  add_arg('--name', dest='name', help='Name for referencing network')
  add_arg('--model', dest='model', help='Select which model type to train')
  add_arg('--nb_train', dest='nb_train',help='Number of training samples',type=int)
  add_arg('--nb_test',  dest='nb_test', help='Number of testing samples', type=int)
  add_arg('--datadir', dest='datadir', help='Directory where data pickle files will be stored')
  add_arg('--raw_datafile', dest='raw_datafile', help='Location of HIGGS.csv')

  args = parser.parse_args()
  return args
