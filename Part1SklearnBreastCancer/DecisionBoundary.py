from DataLoader import load_clean_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

#Settings
target = 'target'
features = ['mean area', 'mean smoothness']
use_subset = False
subset_start = 0
subset_end = 15
test_size = 0.2
random_state = 42
max_iter = 100

#Load the dataset
df = load_clean_data()

# Drop columns and extract features + labels
X = df[features]
y = df[target]

# Select only the rows for experimentation
if use_subset:
    X = X.iloc[subset_start:subset_end]
    y = y.iloc[subset_start:subset_end]

#Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

#Fit the model
reg = LogisticRegression(max_iter=max_iter)
reg.fit(X_train, y_train)

#Create meshgrid
x_margin = (X[features[0]].max() - X[features[0]].min()) * 0.1
y_margin = (X[features[1]].max() - X[features[1]].min()) * 0.1

x_min, x_max = X[features[0]].min() - x_margin, X[features[0]].max() + x_margin
y_min, y_max = X[features[1]].min() - y_margin, X[features[1]].max() + y_margin
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

#Add predictions of datapoints
Z = reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#Plot decision boundary
plt.rcParams['figure.figsize']=(10,10)
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.4)

#Draw model
scatter = plt.scatter(X[features[0]], X[features[1]], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title("Beslutningsgrænse (Logistisk Regression)")
plt.legend(*scatter.legend_elements(), title="Klasse")
plt.show()