import torch 
import torch.nn as nn      

class QKDGatekeeperLSTM(nn.Module):
    def __init__(self,input_size = 3, hidden_dim = 64, num_layers = 2, dropout = 0.2):
        super(QKDGatekeeperLSTM, self).__init__()
        
        # 1. Multi-layer LSTM
        # batch_first=True expects tensors of shape (batch_size, sequence_length, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 2. Fully connected classification head
        self.fc = nn.Linear(hidden_dim, 1)
        
        # 3. Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self,x):
        
        lstm_out,hidden_state= self.lstm(x)
        last_timestep_out = lstm_out[:, -1, :]
        out = self.fc(last_timestep_out)
        prob = self.sigmoid(out)
        
        return prob.squeeze(-1)
    
    
if __name__ == '__main__':
        
    model = QKDGatekeeperLSTM()
    dummyX = torch.randn(64,50,3)
    output = model(dummyX)
    print(output.shape)
    print(output[:5])
            