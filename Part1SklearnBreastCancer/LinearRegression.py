from DataLoader import load_clean_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Settings
target = 'mean radius'
dropColumns = [target]
use_subset = True
subset_start = 0
subset_end = 15
test_size = 0.2
random_state = 42

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
reg = LinearRegression()
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

#Make and print a results table
results_table = pd.DataFrame(data=X_test, columns=X_test.columns)
results_table['Predicted Score'] = y_pred
results_table['Actual Score'] = y_test
results_table['Error'] = results_table['Actual Score']-results_table['Predicted Score']
results_table['Error Squared'] = results_table['Error']* results_table['Error']

print(results_table)