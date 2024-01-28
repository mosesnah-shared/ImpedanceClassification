import torch
import torch.nn    as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Sklearn kit
from sklearn.metrics import accuracy_score
from networks import BinaryClassifier

# Others
import numpy             as np
import matplotlib.pyplot as plt

# Local Libraries
from utils import data_to_dict, set_seeds

if __name__ == "__main__":
    
    # Types of shapes
    idx    = 2
    shapes = [ "cylinder", "hex", "square", "triangle" ]
    shape  = shapes[ idx ]

    # We need to set this to re-generate the code
    set_seeds( )

    # Read the file and parse
    file_dir = "./data/set1/parameters_" + shapes[ idx ] + ".txt"
    data_raw = data_to_dict( file_dir )

    # Parameters for the impedances, stiffness (6) and damping ratio (2)
    desired_keys = ['k_x', 'k_y', 'k_z', 'k_A', 'k_B', 'k_C', 'damp_t', 'damp_r']

    # Extracting out the values for impedances
    imp_vals = { key: data_raw[ key ] for key in desired_keys if key in data_raw }

    # Extracting out the values for the 9th one, which is a boolean array for success trials.
    is_succ = data_raw[ 'success' ]

    # Also, normalize the data from 0 to 1
    # Assuming you have the min and max values for each key
    min_values = {'k_x':   50, 'k_y':   50, 'k_z':   50, 'k_A':   5, 'k_B':   5, 'k_C':   5, 'damp_t': 0.1, 'damp_r': 0.1 }
    max_values = {'k_x': 1000, 'k_y': 1000, 'k_z': 1000, 'k_A': 200, 'k_B': 200, 'k_C': 200, 'damp_t': 1.0, 'damp_r': 1.0 }

    # Normalize the data in filtered_data_dict
    imp_vals_norm = { key: [ ( value - min_values[ key ] ) / ( max_values[ key ] - min_values[ key ]) for value in imp_vals[ key ] ] for key in imp_vals }

    # Extract the arrays for the 8 keys
    data_arrays = [ imp_vals_norm[ key ] for key in desired_keys ]  

    # Use axis=0 if you need to stack them horizontally
    # N x 8 array
    stacked_data = np.stack( data_arrays, axis = 1 )  

    # Ensure the dtype is correct, float32 is commonly used
    X_tensor = torch.tensor( stacked_data, dtype=torch.float32 )  

    # Save the arrays as integer, 0 (FAIL) or 1 (SUCCESS)
    n_array = [ 1 if x else 0 for x in is_succ ]
    y_tensor = torch.tensor( n_array, dtype=torch.float32 ).view(-1, 1) 

    # Define the size of the splits
    # Len get the rows of the array, which is for this case N
    total_size = len( X_tensor ) 
    train_size = int( 0.7 * total_size )
    test_size = total_size - train_size

    # Create the dataset and split it
    dataset = TensorDataset( X_tensor, y_tensor )

    # Split the data to (1) training, (2) validation and (3) test data
    train_data, test_data = random_split(   dataset, [ train_size, test_size ] )

    # Create DataLoaders for each set
    batch_size = 64
    train_loader = DataLoader( train_data, batch_size = batch_size, shuffle = True  )
    test_loader  = DataLoader(  test_data, batch_size = batch_size, shuffle = False )


    # Initialize the model, loss function, and optimizer
    model     = BinaryClassifier( )
    criterion = nn.BCELoss( )
    optimizer = optim.Adam( model.parameters( ), lr = 1e-5 )

    # Training the model
    num_epochs = 2**11
    train_losses = []

    for epoch in range( num_epochs ):

        # Training mode
        model.train( )

        train_loss = 0
        for X_batch, Y_batch in train_loader:
            # Initialize gradient
            optimizer.zero_grad( )
            Y_pred = model( X_batch )
            loss = criterion( Y_pred, Y_batch )
            loss.backward( )
            optimizer.step( )
            train_loss += loss.item( )

        train_losses.append( train_loss / len( train_loader ) )
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss / len(train_loader):.4f}')

    # Plotting the loss values
    plt.plot(train_losses, label='Training Loss')
    plt.title('Loss Value vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.show()

    # Testing the model
    model.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad( ):
        for X_batch, Y_batch in test_loader:
            Y_pred = model( X_batch )
            y_pred_list.append(  Y_pred.numpy( ) )
            y_true_list.append( Y_batch.numpy( ) )

    y_pred_list = np.concatenate( y_pred_list )
    y_true_list = np.concatenate( y_true_list )

    # Calculate accuracy
    accuracy = accuracy_score( y_true_list, np.round( y_pred_list ) )
    print(f'Accuracy on test set: {accuracy * 100:.2f}%')

    # Save the model
    model_path = './models/binary_classifier_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')