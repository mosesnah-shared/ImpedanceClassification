# Import Libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing   import MinMaxScaler

from networks import BinaryClassificationNN

# Read the txt file to parse the dataset
def read_file_to_dict( file_path, data_dict ):

    with open( file_path, 'r' ) as file:
        # Read the first line to get the keys
        keys = file.readline( ).strip( ).split( '\t' )
        
        # Initialize the dictionary keys if it's empty
        if not data_dict:
            for key in keys:
                data_dict[ key ] = []
        
        # Read the subsequent lines to get the values
        for line in file:
            values = line.strip( ).split( '\t' )
            for i, value in enumerate( values ):
                if i < 8:
                    data_dict[ keys[ i ] ].append( float( value ) )
                else:
                    data_dict[ keys[ i ] ].append( value.lower( ) == 'true' )
        
        return data_dict
    


if __name__ == "__main__":

    # Read the dataset from the folder
    folder = "data/set1"
    files  = [ "parameters_cylinder.txt", "parameters_hex.txt", "parameters_square.txt", "parameters_triangle.txt" ]

    # Initialize dictionary and create key-value pair
    all_data = {}

    for file in files:
        file_path = os.path.join( folder, file )
        all_data  = read_file_to_dict( file_path, all_data )
    
    # Using this all_data, train a binary classifier
    X = np.array( [ all_data[ key ] for key in list( all_data.keys( ) )[ :8 ] ] ).T
    y = np.array(   all_data[ 'success'], dtype=np.float32)

    # Standardize the features
    # The min/max values of   k_x	  k_y,	  k_z,	 k_A,   k_B,   k_C, damp_t,	 damp_r
    min_vals = np.array( [   50.0,   50.0,   50.0,   5.0,   5.0,   5.0,    0.1,    0.1 ] )
    max_vals = np.array( [ 1000.0, 1000.0, 1000.0, 200.0, 200.0, 200.0,    1.0,    1.0 ] )

    # Initialize MinMaxScaler with known min and max values
    scaler = MinMaxScaler()
    scaler.fit( X )  # This is necessary to initialize the scaler

    # Manually set the min_ and scale_ attributes
    scaler.min_   = -min_vals / ( max_vals - min_vals )
    scaler.scale_ =         1 / ( max_vals - min_vals )

    # Split the data into training and testing sets
    # Do 70% (train) vs. 30% (test) divide
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state = 42 )
    X_train = scaler.transform( X_train )
    X_test  = scaler.transform( X_test  )

    # Convert data to PyTorch tensors
    X_train = torch.tensor( X_train, dtype = torch.float32 )
    y_train = torch.tensor( y_train, dtype = torch.float32 ).unsqueeze( 1 )
    X_test  = torch.tensor(  X_test, dtype = torch.float32 )
    y_test  = torch.tensor(  y_test, dtype = torch.float32 ).unsqueeze( 1 )

    # Initialize the model, loss function, and optimizer
    model = BinaryClassificationNN( input_size = 8 )
    criterion = nn.BCELoss( )
    optimizer = optim.Adam( model.parameters(), lr=1e-4 )

    # Train the model with cross-validation
    # Divide the training dataset to 3 segments, run training
    kfold = KFold( n_splits=3, shuffle=True, random_state=42 )

    # Iterate over the fold, which is again divided into training and validation set. 
    for fold, ( train_idx, val_idx ) in enumerate( kfold.split( X_train ) ):
        print( f'Fold { fold + 1 }' )
        X_train_fold, X_val_fold = X_train[ train_idx ], X_train[ val_idx ]
        y_train_fold, y_val_fold = y_train[ train_idx ], y_train[ val_idx ]

        # Train the model
        for epoch in range( 2**12 ): 
            model.train()
            optimizer.zero_grad()
            outputs = model( X_train_fold )
            loss    = criterion( outputs, y_train_fold )
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model( X_val_fold )
                val_loss    = criterion( val_outputs, y_val_fold )

            if epoch % 10 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss.item():.6f}, Validation Loss: {val_loss.item():.6f}')

    # Training complete!
    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad( ):
        test_outputs = model( X_test )
        test_loss  = criterion( test_outputs, y_test )
        test_preds = ( test_outputs >= 0.5 ).float( )
        accuracy   = ( test_preds == y_test ).sum( ).item( ) / y_test.size( 0 )

    print( f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy * 100:.2f}%')

    # Save the model
    model_save_path = "models/binary_classification_nn.pth"
    torch.save( model.state_dict(), model_save_path )
    print(f"Model saved to {model_save_path}")