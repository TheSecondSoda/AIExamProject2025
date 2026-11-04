from DataLoader import load_clean_data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, learning_curve
import numpy as np

#Settings
target = 'target'
dropColumns = [target]
use_subset = False
subset_start = 0
subset_end = 15
test_size = 0.2
random_state = 42
max_depth = 4
cv_n_splits = 5

#Load the dataset
df = load_clean_data()

#Drop columns and extract features + labels
X = df.drop(columns=dropColumns)
y = df[target]

#Select only the rows for experimentation
if use_subset:
    X = X.iloc[subset_start:subset_end]
    y = y.iloc[subset_start:subset_end]

#Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

#Fit the model
DT_model = DecisionTreeClassifier(criterion='gini', max_depth=max_depth)
DT_model.fit(X_train, y_train)

y_pred = DT_model.predict(X_test)

def get_metrics():
    #Determine metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    #And cross validation
    k_folds = KFold(n_splits = cv_n_splits)
    cross_scores = cross_val_score(DT_model, X, y, cv = k_folds)

    print(f"Cross-val scores: {cross_scores}")

def get_confusion_matrix(seaborn = False):
    #Make and print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    if seaborn == False:
        #Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
    
    if seaborn == True:
        # Lav heatmap
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malign"], yticklabels=["Benign", "Malign"])
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix (Logistic Regression)")
        plt.show()

def get_plot_tree():
    plt.figure(figsize=(20, 10))
    plot_tree(
        DT_model,
        feature_names=X.columns,
        class_names=["Benign", "Malign"],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title(f"Beslutningstræ (max_depth={max_depth})")
    plt.show()

def get_learning_curve_plot():
    #Get values
    train_sizes, train_scores, test_scores = learning_curve(
        DT_model, X, y, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_score_mean = np.mean(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)

    #Make plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_score_mean, label='Decision Tree - Train', linestyle='--', marker='o')
    plt.plot(train_sizes, test_score_mean, label='Decision Tree - Test', marker='o')

    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curves of DecisionTree")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #get_confusion_matrix(seaborn = False)
    #get_plot_tree()
    get_metrics()
    #get_learning_curve_plot()