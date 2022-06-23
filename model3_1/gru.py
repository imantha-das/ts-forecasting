import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, output_size, dropout_prob):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size 
        self.layer_size = layer_size 

        # GRU Layer
        # (N, L, H_in), (num_layer, N, H_out) ---> (N, L, H_out), (num_layer, N, H_out)
        self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = layer_size, batch_first = True, dropout = dropout_prob)
        # FCC
        # (*, H_in) ---> (*, H_out)
        self.fc = nn.Linear(in_features=hidden_size, out_features = 1)
        
    def forward(self, x):
        # initialize hidden state
        #(numlayers. N, H_out)
        h0 = torch.zeros(self.layer_size,x.shape[0], self.hidden_size).requires_grad_()
        # Forward pass
        out, hn = self.gru(x, h0.detach())
        out = out[:, -1, :] 
        out = self.fc(out)
        return out