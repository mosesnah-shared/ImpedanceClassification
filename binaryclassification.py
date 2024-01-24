import torch
import numpy    as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import data_to_dict

# Define the network architecture
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer_1 = nn.Linear(8, 256) # First hidden layer
        self.layer_2 = nn.Linear(256, 32) # Second hidden layer
        self.layer_out = nn.Linear(32, 1) # Output layer
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.layer_out(x)
        x = self.sigmoid(x)  # Sigmoid activation for binary classification
        return x


# For the first example, read the square data
# file_dir = "./data/set1/parameters_square.txt"

file_dir = "./data/set1/parameters_hex.txt"
data_raw = data_to_dict( file_dir )

# Desired keys
desired_keys = ['k_x', 'k_y', 'k_z', 'k_A', 'k_B', 'k_C', 'damp_t', 'damp_r']

# Extracting only the desired keys and their corresponding values
imp_vals = { key: data_raw[ key ] for key in desired_keys if key in data_raw }

# Extracting only success trials
is_succ = data_raw[ 'success' ]

# Also, normalize the data from 0 to 1
# Assuming you have the min and max values for each key
min_values = {'k_x':   50, 'k_y':   50, 'k_z':   50, 'k_A':   5, 'k_B':   5, 'k_C':   5, 'damp_t': 0.1, 'damp_r': 0.1 }
max_values = {'k_x': 1000, 'k_y': 1000, 'k_z': 1000, 'k_A': 200, 'k_B': 200, 'k_C': 200, 'damp_t': 1.0, 'damp_r': 1.0 }

# Normalize the data in filtered_data_dict
imp_vals_norm = { key: [ ( value - min_values[ key ] ) / ( max_values[ key ] - min_values[ key ]) for value in imp_vals[ key ] ] for key in imp_vals }

# Extract the arrays for the 8 keys
data_arrays = [ imp_vals_norm[ key ] for key in desired_keys ]  
stacked_data = np.stack(data_arrays, axis=1)  # Use axis=0 if you need to stack them vertically
X_train = torch.tensor(stacked_data, dtype=torch.float32)  # Ensure the dtype is correct, float32 is commonly used

# Stack the arrays to create a 2D array if you have multiple label arrays
# If you only have one array of labels, you might skip this step
n_array = [1 if x else 0 for x in is_succ ]

y_train = torch.tensor( n_array, dtype=torch.float32 ).view(-1, 1) 


# Initialize the model
model = BinaryClassifier()

# Specify the loss function
criterion = nn.BCELoss()  # Binary Cross Entropy Loss

# Specify the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model (skeleton code)
num_epochs = 2000

loss_values = []


for epoch in range(num_epochs):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(X_train)

    # Compute loss
    loss = criterion(y_pred, y_train)

    # Save the loss
    loss_values.append(loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
# Plotting the loss values
plt.plot(loss_values)
plt.title('Loss Value vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.show()		