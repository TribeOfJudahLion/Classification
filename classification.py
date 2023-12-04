import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve

input_file = 'car.data.txt'

# Reading and preprocessing the data
X = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data)

X = np.array(X)
label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    label_encoder.append(preprocessing.LabelEncoder())
    X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 8, 15]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=7), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters and best score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

# Building the best classifier
best_classifier = grid_search.best_estimator_

# Feature Importance Analysis
importances = best_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]})")

# Predict on test data
accuracy = best_classifier.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Learning Curve
train_sizes, train_scores, validation_scores = learning_curve(
    best_classifier, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)

plt.figure()
plt.plot(train_sizes, 100 * np.mean(train_scores, axis=1), color='black')
plt.title('Learning curve')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.show()
