# Import pip Libraries
import torch
import numpy    as np
from networks import BinaryClassificationNN
from sklearn.preprocessing import MinMaxScaler

# [WARNING] [Moses C. Nah] [2024.05.17]
    #   Care is required when we import the Neural Network, as the 
    #   Neural Network Architecture of the trained and imported models must be identical.
    #   Currently, we used the Neural Network Architecture of 8x64x64x1.
def load_model( model_path, input_size ):
    model = BinaryClassificationNN( input_size = input_size )
    model.load_state_dict( torch.load( model_path ) )

    # We are NOT training the Neural Network.
    # Hence, must tearn on the eval( ) mode
    model.eval()
    return model

# The input from the user must be within the min/max value range
def validate_input( input_data ):
    # The min/max values of   k_x	  k_y,	  k_z,	 k_A,   k_B,   k_C, damp_t,	 damp_r
    min_vals = np.array( [   50.0,   50.0,   50.0,   5.0,   5.0,   5.0,    0.1,    0.1 ] )
    max_vals = np.array( [ 1000.0, 1000.0, 1000.0, 200.0, 200.0, 200.0,    1.0,    1.0 ] )

    # Check whether the input is valid
    for i, value in enumerate( input_data ):
        if not ( min_vals[ i ] <= value <= max_vals[ i ] ):
            raise ValueError( f"Input value {value:.4f} at index { i } is out of the allowed range [{ min_vals[ i ]:.2f }, { max_vals[ i ]:.2f } ]" )

# Since the Neural Network was trained with normalized data range from 0 to 1, 
# Normalize the data with min/max scale
def normalize_input( input_data ):

    # The min/max values of   k_x	  k_y,	  k_z,	 k_A,   k_B,   k_C, damp_t,	 damp_r
    min_vals = np.array( [   50.0,   50.0,   50.0,   5.0,   5.0,   5.0,    0.1,    0.1 ] )
    max_vals = np.array( [ 1000.0, 1000.0, 1000.0, 200.0, 200.0, 200.0,    1.0,    1.0 ] )

    # Scaling with min/max value 
    scaler = MinMaxScaler()
    scaler.fit( [ min_vals, max_vals ] )
    normalized_data = scaler.transform( [ input_data ] )

    return normalized_data[ 0 ]

# Predicting the output by Binary Classier Neural Network
def predict( model, input_data ):

    # Check whether user input is valid
    validate_input( input_data )

    # If valid input, normalize from 0 to 1 scale
    normalized_data = normalize_input(input_data)

    # Tensorize the input data, which must be done to avoid annoying warning messages
    input_tensor = torch.tensor( np.array( [ normalized_data ] ), dtype = torch.float32 )

    with torch.no_grad():
        output = model( input_tensor )
    return output.item( )

if __name__ == "__main__":
    # Path to the saved model
    # [Note] [Moses C. Nah] [2024.05.17]
    #   We use the Neural Network that is trained without cross validation
    model_path = "models/binary_classification_nn_no_cv1.pth"  

    # Load the model
    model = load_model( model_path, input_size = 8 )

    # Repeat input and prediction process until interrupted
    try:
        while True:
            try:
                # Get input from the user
                input_data = input( "Enter 8 numbers separated by space: " )
                input_data = list( map( float, input_data.split( ) ) )

                # There should be 8 numbers for the input
                if len(input_data) != 8:
                    raise ValueError( "You must enter exactly 8 numbers." )

                try:
                    prediction = predict( model, input_data )
                    result = "Success" if prediction >= 0.5 else "Failure"
                    print( f"Prediction: {result}" )
                except ValueError as e:

                    print( e )

            except ValueError as ve:
                print( ve )
                print("Please try again.")

    except KeyboardInterrupt:
        print( "\nProcess interrupted by user. Exiting..." )