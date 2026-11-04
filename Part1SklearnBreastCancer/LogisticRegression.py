from DataLoader import load_clean_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

#Settings
target = 'target'
dropColumns = [target]
use_subset = False
subset_start = 0
subset_end = 15
test_size = 0.2
random_state = 42
max_iter = 1000
penalty = 'l2'
solver = 'lbfgs'

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

#Fit the model
reg = LogisticRegression(penalty=penalty, solver=solver, max_iter=max_iter)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

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

def get_roc_curve():
    #predict probabilities
    y_pred_proba = reg.predict_proba(X_test)[:,1]

    #ROC-curve
    [fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    #plot ROC-curve
    plt.figure()
    plt.plot(fpr, tpr, color='coral', label = 'ROC Curve with Area Under Curve ='+str(auc_score))
    plt.xlabel('False positive Rate')
    plt.ylabel('True Positive Rate ')
    plt.legend(loc='lower right')
    plt.show()

def plot_logistic_weights(penalties, solvers):
    """
    Trains logistic regression models using different penalties, and plots the model coefficients.

    penalties: List of penalty types (None, 'l1', 'l2')
    solvers: Dictionary mapping each penalty to a solver
    """
    #Save feature names
    feature_names = X.columns

    #Add features to plot
    plt.figure(figsize=(12, 6))
    for penalty in penalties:
        solver = solvers.get(penalty, 'lbfgs') #Defaults to lbfgs, but it's not completely foolproof as lbfgs can't take the l1 penalty
        regmodel = LogisticRegression(penalty=penalty, solver=solver, max_iter=max_iter)
        regmodel.fit(X_train, y_train)
        coefs = regmodel.coef_[0] #The models .coef value is where the model stored all of the learned coefficients, the [0] isn't really needed for a binary classification, but it's here as a note in canse one want's to copy this code for a multiclass classification later
        label = f"Penalty: {penalty or 'None'}"
        plt.plot(range(len(coefs)), coefs, marker='o', label=label)

    #Make and show plot
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.ylabel("Vægtværdi (koefficient)")
    plt.title("Effekten af regulering på modelkoefficienter")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #get_confusion_matrix(seaborn = True)
    #get_metrics()
    #get_roc_curve()
    plot_logistic_weights(penalties = [None, 'l1', 'l2'], solvers = {None: 'lbfgs', 'l1': 'liblinear', 'l2': 'liblinear'})