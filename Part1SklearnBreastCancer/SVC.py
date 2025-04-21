from DataLoader import load_clean_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Settings
target = 'target'
dropColumns = [target]
use_subset = False
subset_start = 0
subset_end = 15
test_size = 0.2
random_state = 42

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

svcs = []
def add_svc_to_list(kernel = 'linear'):
    svc = SVC(kernel=kernel)
    svc.fit(X_train, y_train)
    svcs.append(svc)

def get_accuracy_scores():
    for svc in svcs:
        y_pred = svc.predict(X_test)
        svm_accuracy = accuracy_score(y_test, y_pred)
        print(f"SVM Accuracy ({svc.kernel}): {svm_accuracy:.4f}")

if __name__ == "__main__":
    add_svc_to_list('linear')
    add_svc_to_list('rbf')
    add_svc_to_list('poly')
    get_accuracy_scores()