import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from mpl_toolkits.mplot3d import Axes3D

# Generating a synthetic dataset for binary classification
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, flip_y=0.1, class_sep=1.5, random_state=7)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Normalize the features (important for neural networks)
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

# Define the Neural Network Model
class BinaryClassifier(torch.nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer_1 = torch.nn.Linear(2, 64)  # 2 input features, 4 neurons in the first hidden layer
        self.layer_2 = torch.nn.Linear(64, 32)  # 4 neurons in the second hidden layer
        self.layer_3 = torch.nn.Linear(32, 16)  # 4 neurons in the second hidden layer
        self.layer_4 = torch.nn.Linear(16, 8)  # 4 neurons in the second hidden layer
        self.layer_out = torch.nn.Linear(8, 1)  # 1 output
        
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = torch.sigmoid(self.layer_out(x))  # Sigmoid activation for binary classification
        return x

# Initialize the model
model = BinaryClassifier()

# Loss function and optimizer
criterion = torch.nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# List to store loss values
loss_values = []


# Function to plot decision boundary and loss over epochs
def plot_intermediate_results(X, y, model, epoch, loss_values):
    print(f'Epoch {epoch}: train loss: {loss_values[-1]}')
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    spacing = min(x_max - x_min, y_max - y_min) / 100
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                         np.arange(y_min, y_max, spacing))
    
    data = np.hstack((XX.ravel().reshape(-1, 1), 
                      YY.ravel().reshape(-1, 1)))
    
    # For binary classification, use 0.5 as the threshold.
    db_prob = model(torch.tensor(data, dtype=torch.float32))
    Z = (db_prob >= 0.5).float()
    Z = Z.reshape(XX.shape)
    
    plt.figure(figsize=(12, 8))
    plt.contourf(XX, YY, Z, alpha=0.5, levels=np.linspace(0, 1, 3))
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), edgecolor='k')
    plt.title(f'Decision Boundary at Epoch {epoch}')
    plt.savefig(f'decision_boundary_epoch_{epoch}.png')  # Save figure
    plt.close()  # Close the figure


# Training Loop
for epoch in range(0):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    y_pred = model(X_train)
    
    # Compute Loss
    loss = criterion(y_pred, y_train)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    loss_values.append(loss.item())

    if epoch % 100 == 0:
        print(f'Epoch {epoch}: train loss: {loss.item()}')

    # if epoch % 1000 == 0:
    #     plot_intermediate_results(X, y, model, epoch, loss_values)



# Function to plot decision boundary
def plot_decision_boundary(X, model):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    spacing = min(x_max - x_min, y_max - y_min) / 100
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                         np.arange(y_min, y_max, spacing))
    
    data = np.hstack((XX.ravel().reshape(-1, 1), 
                      YY.ravel().reshape(-1, 1)))
    
    # For binary classification, use 0.5 as the threshold.
    db_prob = model(torch.tensor(data, dtype=torch.float32))
    Z = (db_prob >= 0.5).float()
    Z = Z.reshape(XX.shape)
    
    plt.figure(figsize=(12, 8))
    # plt.contourf(XX, YY, Z, alpha=0.5, levels=np.linspace(0, 1, 3))
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), edgecolor='k')
    
    plt.show()

# Plotting the decision boundary
plot_decision_boundary(X, model)


# Plotting the loss over epochs
# plt.figure(figsize=(10, 5))
# plt.plot(loss_values, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss over Epochs')
# plt.legend()
# plt.grid(True)
# plt.show()