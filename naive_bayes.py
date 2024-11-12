# # Imports
# import numpy as np
# import csv
# import pandas as pd

# # Define class for node in decision tree
# class Node:
#     def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
#         self.feature = feature
#         self.threshold = threshold
#         self.left = left
#         self.right = right
#         self.value = value

#     def is_leaf_node(self):
#         return self.value is not None

# # Function: Calculate entropy
# def calculate_entropy(y):
#     """
#     Calculate the entropy of a list of labels
#     :param y: list of labels
#     :return: entropy
#     """
#     # Initialize entropy
#     entropy = 0
#     # Get unique labels
#     labels = np.unique(y)
#     # Get number of labels
#     n = len(y)
#     # Loop through labels
#     for label in labels:
#         # Calculate probability of label
#         p = np.sum(y == label) / n
#         # Update entropy
#         entropy += p * np.log2(p)
#     # Return entropy
#     return entropy * -1

# # Function: Calculate conditional entropy
# def calculate_conditional_entropy(x):
#     labels = np.unique(x)
#     n = len(x)
#     cond_entropy = 0
#     for label in labels:
#         p = np.sum(x == label) / n
#         cond_entropy += p * calculate_entropy(train_data['label'][x == label])
#     return cond_entropy

# # Function: Discretize using k-means clustering
# def k_means(n):
#     # Initialize centroids randomly
#     centroids = np.random.randn(n, 2)
#     # Initialize clusters
#     clusters = np.zeros(n)
#     # Initialize convergence flag
#     converged = False
#     # Loop until convergence
#     while not converged:
#         # Update clusters
#         for i in range(n):
#             # Calculate distances
#             distances = np.linalg.norm(n[i] - centroids, axis=1)
#             # Update clusters
#             clusters[i] = np.argmin(distances)
#         # Update centroids
#         for i in range(n):
#             # Get points in cluster
#             points = n[clusters == i]
#             # Update centroid
#             centroids[i] = np.mean(points, axis=0)
#         # Check for convergence
#         converged = True
#         for i in range(n):
#             # Get points in cluster
#             points = n[clusters == i]
#             # Calculate new centroid
#             new_centroid = np.mean(points, axis=0)
#             # Check distance
#             if np.linalg.norm(new_centroid - centroids[i]) > 1e-4:
#                 converged = False
#                 break
#     return clusters
# def discretize(X, k):
#     """
#     Discretize a continuous feature using k-means clustering
#     :param X: continuous feature
#     :param k: number of clusters
#     :return: discretized feature
#     """
#     # Initialize k-means object
#     kmeans = k_means(k)
#     # Fit k-means object
#     kmeans.fit(X)
#     # Return discretized feature
#     return kmeans.labels_

# train_data = pd.read_csv('train_data.csv')
# # print(train_data.head())

# label_entropy = calculate_entropy(train_data['label'])

# gain = []
# for column in train_data.columns:
#     if column != 'label':
#         # print (calculate_conditional_entropy(train_data[column]))
#         gain.append((column, label_entropy - calculate_conditional_entropy(train_data[column])))
# print(max(gain, key=lambda x: x[1]))

# # Calcualte gain using the discretized features
# discretized_gain = []
# for column in train_data.columns:
#     if column != 'label':
#         discretized_feature = discretize(train_data[column], 5)
#         discretized_gain.append((column, label_entropy - calculate_conditional_entropy(discretized_feature)))
# print(max(discretized_gain, key=lambda x: x[1]))




import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.feature_stats = {}

    def fit(self, X, y):
        """
        Fit the Naive Bayes model according to the training data.
        X: DataFrame of features
        y: Series of labels (0 for away win, 1 for home win)
        """
        # Calculate class priors
        class_counts = y.value_counts().to_dict()
        total_samples = len(y)
        self.class_priors = {c: class_counts[c] / total_samples for c in class_counts}

        # Calculate mean and variance for each feature by class
        self.feature_stats = {}
        for feature in X.columns:
            self.feature_stats[feature] = {}
            for c in np.unique(y):
                feature_values = X[feature][y == c]
                self.feature_stats[feature][c] = {
                    "mean": feature_values.mean(),
                    "var": feature_values.var() + 1e-6  # add small value to avoid zero variance
                }

    def calculate_likelihood(self, x, mean, var):
        """
        Calculate the Gaussian likelihood of a feature value given mean and variance.
        """
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent

    def predict(self, X):
        """
        Predict the class labels for the given data.
        X: DataFrame of features
        Returns: List of predicted class labels
        """
        y_pred = []
        for _, x in X.iterrows():
            class_probs = {}
            for c in self.class_priors:
                # Start with prior probability of the class
                class_probs[c] = np.log(self.class_priors[c])  # Use log to prevent underflow
                for feature in X.columns:
                    mean = self.feature_stats[feature][c]["mean"]
                    var = self.feature_stats[feature][c]["var"]
                    likelihood = self.calculate_likelihood(x[feature], mean, var)
                    class_probs[c] += np.log(likelihood)  # Sum log probabilities

            # Choose the class with the highest posterior probability
            y_pred.append(max(class_probs, key=class_probs.get))
        return y_pred

# Load training data
train_data = pd.read_csv('train_data.csv')

# Preprocess training data
train_data['home_win'] = (train_data['pts_home_avg5'] > train_data['pts_away_avg5']).astype(int)
y_train = train_data['home_win']  # target variable
X_train = train_data.drop(columns=['home_win'])

# Encode categorical columns
categorical_cols = ['team_abbreviation_home', 'team_abbreviation_away', 'season_type', 'home_wl_pre5', 'away_wl_pre5']
for col in categorical_cols:
    X_train[col] = pd.factorize(X_train[col])[0]

# Train Naive Bayes classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)

# Load validation data
validation_data = pd.read_csv('validation_data.csv')

# Preprocess validation data (apply the same encoding as training data)
for col in categorical_cols:
    validation_data[col] = pd.factorize(validation_data[col])[0]

# Predict and output results
predictions = nb_classifier.predict(validation_data)

# Print each prediction (either 0 or 1) line by line
for prediction in predictions:
    print(prediction)
