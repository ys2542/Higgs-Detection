import time
import logging

from loading.data.load_data import load_data
from experiment_handler.read_args import read_args

# Parse arugments
args = read_args()

# Set up logging
logging_level = logging.INFO
logging.basicConfig(format='%(message)s',level=logging_level)


# Load data
train_X, train_y, test_X, test_y = load_data(args.datadir, args.raw_datafile, args.nb_train, args.nb_test)

# Ensure train, test sizes correct in case requested more samples than available
args.nb_train = train_y.shape[0]
args.nb_test  = test_y.shape[0]

logging.info("{} train samples".format(train_y.shape[0]))
logging.info("{} test samples".format(test_y.shape[0]))

t0 = time.time()

# Choose model and run
if args.model== 'logistic':
  import models.basic.logistic as model
elif args.model== 'random_forest':
  import models.basic.random_forest as model
elif args.model== 'gradient_boosting_tree':
  import models.basic.gradient_boosting_tree as model
elif args.model == 'deep_nn':
  import models.mlps.deep_net.main as model
elif args.model == 'gnn':
  import models.mlps.gnn.main as model
else:
  raise Exception("Model type not recognized or no model selected")
model.train_and_evaluate(train_X, train_y, test_X, test_y)

logging.info("Model took {:.2f} minutes to train".format((time.time()-t0)/60))
