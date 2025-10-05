import torch
import torch.nn as nn
import torch.optim as optim


class ANNModel(nn.Module):
    def __init__(self, input_dim, layer1=32, layer2=48, output_dim=1,
                 drop_rate=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = layer1 
        self.layer2 = layer2

        self.drop_rate = drop_rate
        
        self.fc1 = nn.Linear(self.input_dim, self.layer1, )
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.fc3 = nn.Linear(self.layer2, self.output_dim)

        self.dropout = nn.Dropout(self.drop_rate)
        self.activation = nn.ReLU()
        
    def forward(self, x):
  
        x = self.activation(self.fc1(x))
        x = self.activation(self.dropout(self.fc2(x)))
        x = self.fc3(x)

        return x

    
