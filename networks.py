import torch.nn as nn

# The Neural Network Model for Binary Classification
# The sizes of the hidden layer are chosen arbitrarily.
class BinaryClassificationNN( nn.Module ):
    def __init__( self, input_size ):
        super( BinaryClassificationNN, self ).__init__( )
        self.layer_1   = nn.Linear( input_size, 64 )
        self.layer_2   = nn.Linear(         64, 64 )
        self.layer_out = nn.Linear(         64,  1 )
        self.relu      = nn.ReLU()
        self.sigmoid   = nn.Sigmoid()

    def forward( self, x ):
        x = self.relu(    self.layer_1( x ) )
        x = self.relu(    self.layer_2( x ) )
        x = self.sigmoid( self.layer_out( x ) )
        return x