import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from utils import data_to_dict

if __name__ == "__main__":
    
    # Types of shapes
    idx    = 3
    shapes = [ "cylinder", "hex", "square", "triangle" ]
    b_size = [        64,   64,     512,      64 ]
    # Read the file and parse
    file_dir = "./data/set1/parameters_" + shapes[ idx ] + ".txt"
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
    X_tensor = torch.tensor(stacked_data, dtype=torch.float32)  # Ensure the dtype is correct, float32 is commonly used

    # Stack the arrays to create a 2D array if you have multiple label arrays
    # If you only have one array of labels, you might skip this step
    n_array = [1 if x else 0 for x in is_succ ]

    y_tensor = torch.tensor( n_array, dtype=torch.float32 ).view(-1, 1) 

    # Define the size of the splits
    total_size = len(X_tensor)
    train_size = int(0.7 * total_size)
    valid_size = int(0.2 * total_size)
    test_size = total_size - train_size - valid_size

    # Create the dataset and split it
    dataset = TensorDataset(X_tensor, y_tensor)
    train_data, temp_data = random_split(dataset, [train_size, valid_size + test_size])
    valid_data, test_data = random_split(temp_data, [valid_size, test_size])

    # Create DataLoaders for each set
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Define the network architecture
    class BinaryClassifier(nn.Module):
        def __init__(self):
            super(BinaryClassifier, self).__init__()
            self.layer_1 = nn.Linear( 8, 256)  # Adjust the input dimension
            self.layer_2 = nn.Linear(256, 32)
            self.layer_out = nn.Linear(32, 1)
            
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.relu(self.layer_2(x))
            x = self.layer_out(x)
            x = self.sigmoid(x)
            return x

    # Initialize the model, loss function, and optimizer
    model = BinaryClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training the model
    num_epochs = 2400
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))
        
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in valid_loader:
                Y_pred = model(X_batch)
                loss = criterion(Y_pred, Y_batch)
                valid_loss += loss.item()
        valid_losses.append(valid_loss / len(valid_loader))

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {valid_loss / len(valid_loader):.4f}')

    # Plotting the loss values
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Loss Value vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.show()

    # Testing the model
    model.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            Y_pred = model(X_batch)
            y_pred_list.append(Y_pred.numpy())
            y_true_list.append(Y_batch.numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_true_list = [a.squeeze().tolist() for a in y_true_list]

    # Calculate accuracy
    for i in range( len( y_true_list ) ):
        accuracy = accuracy_score(y_true_list[ i ], np.round(y_pred_list[ i ]))
        print(f'Accuracy on test set: {accuracy * 100:.2f}%')

    # Save the model
    model_path = './models/binary_classifier_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')