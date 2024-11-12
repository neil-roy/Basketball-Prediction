# Imports
import sys
import numpy as np
import pandas as pd

# Define class
class NaiveBayesClassifier:
    # initalize class
    def __init__(self):
        self.class_priors = {}
        self.continuous_variable_stats = {}
        self.categorical_variable_probs = {}
    
    # train the model
    def train(self, X, y, categorical_cols):
        label_counts = y.value_counts().to_dict()
        size = len(y) # 10,000
        self.class_priors = {}
        for c in label_counts:
            self.class_priors[c] = label_counts[c] / size
        
        for c in np.unique(y): # either c = 0 or c = 1
            X_c = X[y == c]
            self.continuous_variable_stats[c] = {}
            self.categorical_variable_probs[c] = {}

            # continuous variables  (easy)
            for variable in X.columns.difference(categorical_cols):
                variable_values = X_c[variable]
                self.continuous_variable_stats[c][variable] = {
                    "mean": variable_values.mean(),
                    "var": variable_values.var() + 1e-6
                }

            # categorical variables (edit later to see I can increase accuracy) (hard)
            for variable in categorical_cols:
                variable_counts = X_c[variable].value_counts()
                total_count_c = len(X_c)
                categories = X[variable].cat.categories
                num_categories = len(categories)
                probs = {}
                for i in categories:
                    count = variable_counts.get(i, 0) + 1
                    prob = count / (total_count_c + num_categories + 1)
                    probs[i] = prob
                self.categorical_variable_probs[c][variable] = {
                    'probs': probs,
                    'total_count_c': total_count_c,
                    'num_categories': num_categories
                }

    # calculate likelihood
    def calculate_likelihood(self, x, mean, var):
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent
    
    # def calculate_likelihood(self, x, mean, var):
    #     # Using logarithms to prevent underflow
    #     exponent = -((x - mean) ** 2) / (2 * var)
    #     return exponent - 0.5 * np.log(2 * np.pi * var)

    
    # Function: predict the input data
    def predict(self, X, categorical_cols):
        predictions = []
        for _, x in X.iterrows():
            probs = {}
            for c in self.class_priors:
                probs[c] = np.log(self.class_priors[c])
                # continuous variables
                for variable in X.columns.difference(categorical_cols):
                    if (variable == 'label'):
                        continue
                    mean = self.continuous_variable_stats[c][variable]["mean"]
                    var = self.continuous_variable_stats[c][variable]["var"]
                    likelihood = self.calculate_likelihood(x[variable], mean, var)
                    probs[c] = probs[c] + np.log(likelihood)
                # categorical variables
                for variable in categorical_cols:
                    prob = self.categorical_variable_probs[c][variable]['probs'].get(x[variable], 1 / (self.categorical_variable_probs[c][variable]['total_count_c'] + self.categorical_variable_probs[c][variable]['num_categories']))
                    probs[c] = probs[c] + np.log(prob)
            predictions.append(max(probs, key=probs.get))
        return predictions

# Main function
if __name__ == "__main__":

    # import data
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    # load training data (and seerate)
    train_data = pd.read_csv(train_file)
    results = train_data['label']
    features = train_data.drop(columns=['label'])

    # load testing data
    test_data = pd.read_csv(test_file)
    # test_results = test_data['label']
    # test_features = test_data.drop(columns=['label'])
    test_features = test_data
    
    # create a list of all the categorical column names
    categorical_cols = ['team_abbreviation_home', 'team_abbreviation_away', 'season_type', 'home_wl_pre5', 'away_wl_pre5']

    # convert to categorical type
    for col in categorical_cols:
        categories = pd.CategoricalDtype(categories=features[col].unique())
        features[col] = features[col].astype(categories)
        test_features[col] = test_features[col].astype(categories)

    # train the model
    model = NaiveBayesClassifier()
    model.train(features, results, categorical_cols)

    # predict the test data
    predictions = model.predict(test_features, categorical_cols)

    # print the results
    for prediction in predictions:
        print(prediction)
    
    # calculate the accuracy (local use only)
    # accuracy = np.mean(predictions == test_results)
    # print(f"Accuracy: {accuracy}")
        
        

# End of file
        

# Testing ideas for improvement in accuracy
# 1. Add more features

# Function: calculate information gain for each feature
    
# def calculate_information_gain(X, y):
#     information_gains = {}
#     for column in X.columns:
#         values = X[column].unique()
#         entropy = 0
#         for value in values:
#             p = len(X[X[column] == value]) / len(X)
#             entropy += p * np.log2(p)
#         information_gains[column] = entropy
#     return information_gains

# gains = calculate_information_gain(features, results)
# # sort gains
# sorted_gains = sorted(gains.items(), key=lambda x: x[1], reverse=True)
# print(sorted_gains)