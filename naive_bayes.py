# Imports
import numpy as np
import csv

# Function: Calculate entropy
def calculate_entropy(y):
    """
    Calculate the entropy of a list of labels
    :param y: list of labels
    :return: entropy
    """
    # Initialize entropy
    entropy = 0
    # Get unique labels
    labels = np.unique(y)
    # Get number of labels
    n = len(y)
    # Loop through labels
    for label in labels:
        # Calculate probability of label
        p = np.sum(y == label) / n
        # Update entropy
        entropy -= p * np.log2(p)
    # Return entropy
    return entropy

with open('train_data.csv', 'r') as file:
    csv_reader = csv.reader(file)
    data = []
    for row in csv_reader:
        data.append(row)



