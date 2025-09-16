import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.utils import softmax  
# import torch.nn.functional as F
from torch_scatter import scatter_add  
from torch.autograd import Variable

class LightweightContext(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, input_dim // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.dropout(out)  # no projection or attention
    
class IterModule(nn.Module):
    def __init__(self, in_channels=1024, processing_steps=0, num_layers=1):
        super(IterModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        self.GlobalContext = LightweightContext(in_channels)
        if processing_steps > 0:
            self.lstm = nn.LSTM(self.out_channels, self.in_channels, num_layers)
            self.lstm1 = nn.Linear(self.in_channels, self.out_channels)
            self.lstm.reset_parameters()
            self.gate_fc = nn.Linear(2 * in_channels, 1)

    def forward(self, x, batch, q_star):
        if self.processing_steps <= 0: 
            return q_star
        
        batch_size = batch.max().item() + 1
        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star

 