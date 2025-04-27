from DataLoader import load_clean_data
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Settings
target = 'target'
dropColumns = [target]
use_subset = False
subset_start = 0
subset_end = 15
n_clusters = 2

#Load the dataset
df = load_clean_data()

#Drop columns and extract features + labels
X = df.drop(columns=dropColumns)
y = df[target]

#Select only the rows for experimentation
if use_subset:
    X = X.iloc[subset_start:subset_end]
    y = y.iloc[subset_start:subset_end]

def Agglomerative_cluster(X, n_clusters=2):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = model.fit_predict(X)

    return cluster_labels

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    cluster_labels = Agglomerative_cluster(X, n_clusters)
    plot_confusion_matrix(y, cluster_labels)
    