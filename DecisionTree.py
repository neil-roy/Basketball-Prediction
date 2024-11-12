import sys
import numpy as np
import pandas as pd

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)
        
        if len(unique_classes) == 1 or depth >= self.max_depth or num_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)
        
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)
        
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left_subtree = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_subtree = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return DecisionTreeNode(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _best_split(self, X, y):
        num_samples, num_features = X.shape
        best_gain = -1
        split_feature, split_threshold = None, None

        for feature_idx in range(num_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature_idx], threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_feature = feature_idx
                    split_threshold = threshold

        return split_feature, split_threshold

    def _information_gain(self, y, X_column, split_threshold):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        num_left, num_right = len(left_idxs), len(right_idxs)
        num_total = len(y)
        left_entropy = self._entropy(y[left_idxs])
        right_entropy = self._entropy(y[right_idxs])
        child_entropy = (num_left / num_total) * left_entropy + (num_right / num_total) * right_entropy

        return parent_entropy - child_entropy

    def _split(self, X_column, split_threshold):
        left_idxs = np.where(X_column <= split_threshold)[0]
        right_idxs = np.where(X_column > split_threshold)[0]
        return left_idxs, right_idxs

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

if __name__ == "__main__":
    # Get the file paths from command line arguments
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    # Load and preprocess the training data
    train_data = pd.read_csv(train_file)
    train_data['label'] = (train_data['pts_home_avg5'] > train_data['pts_away_avg5']).astype(int)
    y_train = train_data['label'].values
    X_train = train_data.drop(columns=['label']).values

    # Encode categorical features
    categorical_cols = ['team_abbreviation_home', 'team_abbreviation_away', 'season_type', 'home_wl_pre5', 'away_wl_pre5']
    train_data[categorical_cols] = train_data[categorical_cols].apply(lambda col: pd.factorize(col)[0])

    # Train the decision tree classifier
    X_train = train_data.drop(columns=['label']).values
    tree = DecisionTreeClassifier(max_depth=10)
    tree.fit(X_train, y_train)

    # Load and preprocess the test data
    test_data = pd.read_csv(test_file)
    test_data[categorical_cols] = test_data[categorical_cols].apply(lambda col: pd.factorize(col)[0])
    X_test = test_data.values

    # Make predictions
    predictions = tree.predict(X_test)

    # Print predictions line by line
    for prediction in predictions:
        print(prediction)

    # compare
    y_test = test_data['label'].values
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy}")