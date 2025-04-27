from DataLoader import load_clean_data
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

def elbow_method(X, max_clusters=10):
    inertias = []
    cluster_range = range(1, max_clusters+1)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, inertias, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

def KMeans_cluster(X, n_clusters = 2):
    kmeans  = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(X)

    return cluster_labels

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    #elbow_method(X)
    cluster_labels = KMeans_cluster(X, 2)
    plot_confusion_matrix(y, cluster_labels)