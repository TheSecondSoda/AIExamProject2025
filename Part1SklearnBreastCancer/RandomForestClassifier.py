from DataLoader import load_clean_data
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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
n_estimators = 10

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

#Make, fit and use a decisionTreeClassifier
tree_model = DecisionTreeClassifier(max_depth=max_depth)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)

#Make, fit and use a randomForestClassifier
forest_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
forest_model.fit(X_train, y_train)
forest_preds = forest_model.predict(X_test)

def get_metrics():
    #Determine metrics
    tree_acc = accuracy_score(y_test, tree_preds)
    forest_acc = accuracy_score(y_test, forest_preds)

    print(f"DecisionTree accuracy: {tree_acc:.4f}")
    print(f"RandomForest accuracy: {forest_acc:.4f}")

def get_learning_curve_plot():
    #Get values
    train_sizes, train_scores, test_scores = learning_curve(
        forest_model, X, y, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_score_mean = np.mean(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)

    #Make plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_score_mean, label='RandomForest Tree - Train', linestyle='--', marker='o')
    plt.plot(train_sizes, test_score_mean, label='RandomForest Tree - Test', marker='o')

    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curves of RandomForest")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #get_metrics()
    get_learning_curve_plot()