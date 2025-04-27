from DataLoader import load_clean_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import numpy as np

#Settings
target = 'target'
dropColumns = [target]
use_subset = False
subset_start = 0
subset_end = 15
test_size = 0.2
random_state = 42
max_depth = 5
n_estimators = 50


#Load the dataset
df = load_clean_data()

# Drop columns and extract features + labels
X = df.drop(columns=dropColumns)
y = df[target]

# Select only the rows for experimentation
if use_subset:
    X = X.iloc[subset_start:subset_end]
    y = y.iloc[subset_start:subset_end]

#Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

#Base-model: DecisionTree
base_model = DecisionTreeClassifier(max_depth=max_depth)

#Baggingclassifier with bootstrap
bac_bootstrap = BaggingClassifier(estimator=base_model, n_estimators=n_estimators, bootstrap=True)
bac_bootstrap.fit(X_train, y_train)
y_pred_bootstrap = bac_bootstrap.predict(X_test)

#Baggingclassifier without bootstrap
bac_no_bootstrap = BaggingClassifier(estimator=base_model, n_estimators=n_estimators, bootstrap=False)
bac_no_bootstrap.fit(X_train, y_train)
y_pred_no_bootstrap = bac_no_bootstrap.predict(X_test)

def get_metrics():
    #Determine metrics
    accuracy_bootstrap = accuracy_score(y_test, y_pred_bootstrap)
    accuracy_no_bootstrap = accuracy_score(y_test, y_pred_no_bootstrap)

    print(f"Accuracy with bootstrap:     {accuracy_bootstrap:.4f}")
    print(f"Accuracy without bootstrap:  {accuracy_no_bootstrap:.4f}")

def get_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    return train_sizes, train_mean, test_mean

if __name__ == "__main__":
    get_metrics()