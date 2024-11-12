import sys
import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.feature_stats = {}

    def fit(self, X, y):
        class_counts = y.value_counts().to_dict()
        total_samples = len(y)
        self.class_priors = {c: class_counts[c] / total_samples for c in class_counts}

        self.feature_stats = {}
        for feature in X.columns:
            self.feature_stats[feature] = {}
            for c in np.unique(y):
                feature_values = X[feature][y == c]
                self.feature_stats[feature][c] = {
                    "mean": feature_values.mean(),
                    "var": feature_values.var() + 1e-6
                }

    def calculate_likelihood(self, x, mean, var):
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent

    def predict(self, X):
        y_pred = []
        for _, x in X.iterrows():
            class_probs = {}
            for c in self.class_priors:
                class_probs[c] = np.log(self.class_priors[c])
                for feature in X.columns:
                    mean = self.feature_stats[feature][c]["mean"]
                    var = self.feature_stats[feature][c]["var"]
                    likelihood = self.calculate_likelihood(x[feature], mean, var)
                    class_probs[c] += np.log(likelihood)
            y_pred.append(max(class_probs, key=class_probs.get))
        return y_pred

if __name__ == "__main__":
    # Read file paths from command line arguments
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    # Load training data
    train_data = pd.read_csv(train_file)
    train_data['home_win'] = (train_data['pts_home_avg5'] > train_data['pts_away_avg5']).astype(int)
    y_train = train_data['home_win']
    X_train = train_data.drop(columns=['home_win'])

    # Encode categorical columns
    categorical_cols = ['team_abbreviation_home', 'team_abbreviation_away', 'season_type', 'home_wl_pre5', 'away_wl_pre5']
    for col in categorical_cols:
        X_train[col] = pd.factorize(X_train[col])[0]

    # Train Naive Bayes classifier
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(X_train, y_train)

    # Load test data (validation data)
    test_data = pd.read_csv(test_file)
    for col in categorical_cols:
        test_data[col] = pd.factorize(test_data[col])[0]

    # Predict and output results
    predictions = nb_classifier.predict(test_data)
    for prediction in predictions:
        print(prediction)
