from DataLoader import load_clean_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#Settings
target = 'target'
dropColumns = [target]
test_size = 0.2
random_state = 42
learningrate = 0.01
epochs = 1000

#Load the dataset
df = load_clean_data()

#Drop columns and extract features + labels
X = df.drop(columns=dropColumns)
y = df[target]

#Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

#Convert to pytorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

#Define the model
model = nn.Sequential(
    nn.Linear(X.shape[1], 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

#Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learningrate)

#Training loop
losses = []
accuracy = []
for epoch in range(epochs):
    preds = model(X_train_tensor)
    loss = criterion(preds, y_train_tensor)
    losses.append(loss.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pred_labels = (preds > 0.5).float()
    matches = (pred_labels == y_train_tensor).float()
    matchesNumeric = matches.float()
    accuracyPct = 100*torch.mean(matches)
    accuracy.append( accuracyPct )

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

#Model accuracy and losses
fig,ax = plt.subplots(1,2,figsize=(13,4))
ax[0].plot(losses)
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('epoch')
ax[0].set_title('Losses')
ax[1].plot(accuracy)
ax[1].set_ylabel('accuracy')
ax[1].set_xlabel('epoch')
ax[1].set_title('Accuracy')
plt.show()

#Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_labels = (y_pred > 0.5).int().squeeze()

accuracy = accuracy_score(y_test, y_pred_labels.numpy())
print(f"PyTorch NN Accuracy: {accuracy:.4f}")