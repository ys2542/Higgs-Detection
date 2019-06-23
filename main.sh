#!/bin/bash

# Run model through this script
# Modify variables to run different models
# Run 'script/main.py -h' for list of implemented arguments

NAME="test"
MODEL="deep_nn"
NB_TRAIN=100000
NB_TEST=10000
PROCESSED_DATADIR="/home/nicholas/nyuDrive/ml_project/data"
RAW_DATAFILE="/home/nicholas/Downloads/HIGGS.csv"

# Collect arguments from ^^
PY_ARGS="--name $NAME --model $MODEL --nb_train $NB_TRAIN --nb_test $NB_TEST --datadir $PROCESSED_DATADIR --raw_datafile $RAW_DATAFILE"
# Run model
python3 script/main.py $PY_ARGS
