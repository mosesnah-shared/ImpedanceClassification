# Import pip Libraries
import os
import numpy as np
import torch
import torch.nn    as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing   import MinMaxScaler

# Import local Library, mostly the Neural Network
from networks import BinaryClassificationNN

# Function to read and parsing the txt file 
def read_file_to_dict( file_path, data_dict ):

    with open( file_path, 'r' ) as file:
        # Read the first line to get the keys
        # There are 10 keys where the last one is ignored:
        #   k_x k_y k_z k_A k_B k_C damp_t damp_r success partialSuccess	
        keys = file.readline( ).strip( ).split( '\t' )
        
        # Initialize the dictionary keys if one passes an empty dictionary key.
        if not data_dict:
            for key in keys:
                data_dict[ key ] = [ ]
        
        # Read the subsequent lines to get the values
        for line in file:
            values = line.strip( ).split( '\t' )
            for i, value in enumerate( values ):
                # The first 8 values are float
                if i < 8:
                    data_dict[ keys[ i ] ].append( float( value ) )
                # The last 2 values are true/false (boolean)
                else:
                    data_dict[ keys[ i ] ].append( value.lower( ) == 'true' )
        
        return data_dict
    


if __name__ == "__main__":

    # Flag to turn on cross validation or not
    # [Note] [Moses C. Nah] [2024.05.17]
    #   I personally feel that the dataset is not too large to include cross validation.
    #   Hence, it is better to keep this as False
    turn_on_cross_valid = False

    # The dataset txt files
    folder = "data/set1"
    files  = [ "parameters_cylinder.txt", "parameters_hex.txt", "parameters_square_new.txt", "parameters_triangle.txt" ]

    # Initialize dictionary and read the keys and corresponding values. 
    all_data = { }

    for file in files:
        file_path = os.path.join( folder, file )
        all_data  = read_file_to_dict( file_path, all_data )
    
    # Train the binary classifier Neural Network from all_data dictionary
    # X is a 8-dimensional array, Y is output array.
    X = np.array( [ all_data[ key ] for key in list( all_data.keys( ) )[ :8 ] ] ).T
    y = np.array(   all_data[ 'success' ], dtype=np.float32 )

    # The min/max values of   k_x	  k_y,	  k_z,	 k_A,   k_B,   k_C, damp_t,	 damp_r
    min_vals = np.array( [   50.0,   50.0,   50.0,   5.0,   5.0,   5.0,    0.1,    0.1 ] )
    max_vals = np.array( [ 1000.0, 1000.0, 1000.0, 200.0, 200.0, 200.0,    1.0,    1.0 ] )

    # Initialize MinMaxScaler with known min and max values
    # This is required to normalize the features from 0 to 1.
    scaler = MinMaxScaler( )
    scaler.fit( X )  # This is necessary to initialize the scaler

    # Manually set the min_ and scale_ attributes
    # [Note] [Moses C. Nah] [2024.05.17]
    #    Although we keep the code as it is, there will be a easier/fancier way to do this. 
    scaler.min_   = -min_vals / ( max_vals - min_vals )
    scaler.scale_ =         1 / ( max_vals - min_vals )

    # Split the data into training and testing sets
    # Dividing the dataset to 70% (train) and 30% (test)
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state = 42 )
    X_train = scaler.transform( X_train )
    X_test  = scaler.transform( X_test  )

    # Convert data to PyTorch tensors
    X_train = torch.tensor( X_train, dtype = torch.float32 )
    y_train = torch.tensor( y_train, dtype = torch.float32 ).unsqueeze( 1 )
    X_test  = torch.tensor(  X_test, dtype = torch.float32 )
    y_test  = torch.tensor(  y_test, dtype = torch.float32 ).unsqueeze( 1 )

    # Initialize the Neural Network model, loss function (Binary Cross Entropy), and optimizer (ADAM)
    model     = BinaryClassificationNN( input_size = 8 )
    criterion = nn.BCELoss( )
    optimizer = optim.Adam( model.parameters(), lr=1e-4 )

    # Initialize TensorBoard writer, for Data Analysis
    layout = {
        "Loss": {
            "Loss": ["Multiline", ["loss/train", "loss/validation" ] ],
        },
    }
    writer = SummaryWriter( 'runs/experiment8_all_new' )
    writer.add_custom_scalars( layout )

    # Set the number of epochs
    # [Note] [Moses C. Nah] [2024.05.16]
    #    Empirically, I found out that after 2**13 iterations with learning rate 1e-4 (refer to ADAM above)
    #    The validation loss increases whereas the training loss decreases.
    #    This is prominently known as an indication of overfitting. Hence, set the epoch as 2**13.
    Nepoch = 2**13

    if turn_on_cross_valid:
        # Train the model with cross-validation
        # Divide the training dataset into 3 segments, run training
        kfold = KFold( n_splits = 3, shuffle = True, random_state = 42 )

        # Iterate over the fold, which is again divided into training and validation set.
        for fold, ( train_idx, val_idx ) in enumerate( kfold.split( X_train ) ):

            # The number of fold we are currently in
            print( f'Fold {fold + 1}' )
            X_train_fold, X_val_fold = X_train[ train_idx ], X_train[ val_idx ]
            y_train_fold, y_val_fold = y_train[ train_idx ], y_train[ val_idx ]

            # Train the model
            for epoch in range( Nepoch ):
                model.train()
                optimizer.zero_grad()

                # Calculate the training loss
                outputs = model( X_train_fold )
                loss    = criterion( outputs, y_train_fold )

                # Update the weights of the Neural Network by Backpropagation
                loss.backward( )
                optimizer.step( )
                model.eval()

                # Calculate the validation loss
                with torch.no_grad():
                    val_outputs = model( X_val_fold )
                    val_loss    = criterion( val_outputs, y_val_fold )

                # Print out the progress, although can be seen on realtime via TensorBoard
                if epoch % 10 == 0:
                    print(f'Epoch {epoch + 1}, Loss: {loss.item():.6f}, Validation Loss: {val_loss.item():.6f}')

                # Log the loss at TensorBoard
                writer.add_scalars('runs_split', {f'Loss/train_fold{fold}': loss.item(), f'Loss/validation_fold{fold}': val_loss.item()}, epoch + 1)

        print( "Training completed with cross-validation." )

        # Save the trained Neural Network model
        model_save_path = "models/binary_classification_all_nn_cv.pth"

    else:
        # Train the model without cross-validation
        for epoch in range( Nepoch ):
            model.train()
            optimizer.zero_grad()

            # Calculate the training loss
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            # Update the weights of the Neural Network by Backpropagation
            loss.backward()
            optimizer.step()
            model.eval()

            # Calculate the validation loss
            with torch.no_grad():
                val_outputs = model( X_test )
                val_loss    = criterion( val_outputs, y_test )

            if epoch % 10 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss.item():.6f}, Validation Loss: {val_loss.item():.6f}')

            # Log the loss
            writer.add_scalars('runs', {'Loss/train': loss.item(), 'Loss/validation': val_loss.item()}, epoch + 1)

        print( "Training completed without cross-validation." )
        model_save_path = "models/binary_classification_all_nn_no_cv.pth"


    # Training complete!
    # Evaluate the model on the leftover (30%) test set.
    model.eval()

    with torch.no_grad( ):
        test_outputs = model( X_test )
        test_loss    = criterion( test_outputs, y_test )

        # If output is over 0.5, then predict success!
        test_preds   = ( test_outputs >= 0.5 ).float( ) 
        accuracy     = ( test_preds == y_test ).sum( ).item( ) / y_test.size( 0 )

    print( f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy * 100:.2f}%')

    # Save the model
    torch.save( model.state_dict(), model_save_path )
    print(f"Model saved to {model_save_path}")
