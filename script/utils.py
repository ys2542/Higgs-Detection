import os
import pickle

def save_data(save_data, filepath):
  if not os.path.exists(os.path.dirname(filepath)):
    try:
      os.makedirs(os.path.dirname(filepath))
    except:
      raise
  with open(filepath, 'wb') as outfile:
    pickle.dump(save_data, outfile)
