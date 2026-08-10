import torch 
import torch.nn as nn      

class QKDGatekeeperLSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) neural network that detects eavesdropping
    by looking at channel parameters and seeing if photons have been 
    intercepted and resent
    """
    def __init__(self,input_size = 3, hidden_dim = 64, num_layers = 2, dropout = 0.2):
        super(QKDGatekeeperLSTM, self).__init__()
        

        self.lstm = nn.LSTM(
            input_size=input_size, # features per timestamp = 3
            hidden_size=hidden_dim, # nodes to capture temporal patterns
            num_layers=num_layers,# number of lstm layers for deeper feature extraction
            batch_first=True, # makes sure input tensors are in right format
            dropout=dropout if num_layers > 1 else 0 # regularasation to avoid overfitting
        )
        
        # linear layer to map hidden state to singular output
        self.fc = nn.Linear(hidden_dim, 1)
        
        # sigmoid activation to turn ouput into probability
        self.sigmoid = nn.Sigmoid()
        
    # defines the forward pass
    def forward(self,x):

        # pass the input through the layers where lstm_out has the ouput features 
        # for each time step
        lstm_out,hidden_state= self.lstm(x)

        # extract last ouput from last timestep
        last_timestep_out = lstm_out[:, -1, :]
        out = self.fc(last_timestep_out)

        #convert ouput to probability
        prob = self.sigmoid(out)

        # return so shape is (batchsize,)
        return prob.squeeze(-1)
    

    
if __name__ == '__main__':
        
    model = QKDGatekeeperLSTM()
    dummyX = torch.randn(64,50,3)
    output = model(dummyX)
    print(output.shape)
    print(output[:5])
            