import torch 
import torch.nn as nn 

# ------------------------------------------------------------------------------
# LSTM Model
# ------------------------------------------------------------------------------

class LSTM(nn.Module):
    """
    Desc : LSTM Model
    Layers
        LSTM (n layers) -> FC
    """
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, output_size:int = 1,dropout_prob:float = 0.2):
        """
        Inputs
            input_size : number of features
            hidden_size : number of hidden states 
            num_layer : The number of lstm-layers stacked
            output_size : The next value in the sequence
        Outputs
            PyTorch LSTM model
        """
        super(LSTM, self).__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, dropout = dropout_prob)
        self.fc = nn.Linear(in_features = hidden_size, out_features = output_size)

    def forward(self, X):
        """
        Desc : Forward pass in neural network
        Inputs
            X : features
        Outputs
            out : next value in sequence 
        """
        #(num_layers, N, H_out),(num_layers, N, H_cell)
        h0 = torch.zeros(self.num_layers, X.shape[0],self.hidden_size)
        c0 = torch.zeros(self.num_layers, X.shape[0],self.hidden_size)

        # (N,L,H_in),(num_layers, N, H_out),(num_layers, N, H_cell) -> (N,L,H_out),(num_layer, N, H_out),(num_layers,N,H_out)
        out,(hn,cn) = self.lstm(X, (h0,c0))
        # (N,L,H_out) <- However we require last sequence 
        out = out[:,-1,:]
        # (*,H_in)
        out = self.fc(out)

        return out, (hn,cn)
        
