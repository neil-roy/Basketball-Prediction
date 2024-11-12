#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 train_data.csv test_data.csv"
  exit 1
fi

# Assign arguments to variables
TRAIN_DATA=$1
TEST_DATA=$2

# Run the Python script with the provided arguments
python3 DecisionTree.py "$TRAIN_DATA" "$TEST_DATA"