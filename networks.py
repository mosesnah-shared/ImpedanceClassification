
from torch import nn
import torch.nn.init as init
# Define the network architecture
class BinaryClassifier( nn.Module ):
    def __init__(self):
        super( BinaryClassifier, self ).__init__()

        # Adjust the input dimension
        self.layer_1   = nn.Linear(   8, 256 )  
        self.layer_2   = nn.Linear( 256,  32 )
        self.layer_out = nn.Linear(  32,   1 )
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.relu( self.layer_1( inputs ) )
        x = self.relu( self.layer_2( x ) )
        x = self.layer_out( x ) 
        x = self.sigmoid( x )
        return x
    
    def _initialize_weights(self):
        # Initialize weights for the linear layers
        init.xavier_uniform_(self.layer_1.weight)
        init.xavier_uniform_(self.layer_2.weight)
        init.xavier_uniform_(self.layer_out.weight)
        
        # Initialize biases to zero
        init.zeros_(self.layer_1.bias)
        init.zeros_(self.layer_2.bias)    
        init.zeros_(self.layer_out.bias)    

class Autoencoder( nn.Module ):
    def __init__( self ):
        super( Autoencoder, self ).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(   8, 256 ),
            nn.ReLU( True ),
            nn.Linear( 256,  64 ),
            nn.ReLU( True ),
            nn.Linear(  64,  32 ),
            nn.ReLU( True ),
            nn.Linear(  32,  3 ) )
        # Latent space representation        
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(   3,  32 ),
            nn.ReLU( True ),
            nn.Linear(  32,  64 ),
            nn.ReLU( True ),
            nn.Linear(  64, 256 ),
            nn.ReLU( True ),
            nn.Linear( 256,   8 ),
            nn.Sigmoid( )  )
        # Using sigmoid for the last layer if the input data is normalized between 0 and 1

    def forward( self, x ):
        x = self.encoder( x )
        x = self.decoder( x )
        return x
    
    def encode(self, x):
        return self.encoder(x)    

