import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch

class QKDSequenceDataset(Dataset):
    # dataset for QKD telemetry
    # uses look back sequences that will be used for LSTM network
    def __init__(self, df, window_size=50, scaler=None, is_train=True):
        # number of timesteps used for prediction
        self.window_size = window_size
        
        # extract features and ground truths
        X = df[["qber", "jitter", "dark_counts"]].to_numpy()
        y = df["attack_label"].to_numpy()

        
        if is_train:
            # fit training set to new standard scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else: 
            # if test set use scaler from the training set
            self.scaler = scaler
            X_scaled = self.scaler.transform(X)
            
        self.X_seq = []
        self.y_seq = []

        # slide window accross dataset
        for i in range(0, len(X_scaled) - window_size):
            self.X_seq.append(X_scaled[i : i + window_size])
            # last timestep assigned a label as target to predict
            self.y_seq.append(y[i + window_size - 1])
            
        # convert lists to pytorch tensors 
        self.X_seq = torch.tensor(np.array(self.X_seq), dtype=torch.float32)
        self.y_seq = torch.tensor(np.array(self.y_seq), dtype=torch.float32)
    
    def __len__(self):
        # return number of sequence windows used
        return len(self.X_seq)
    
    def __getitem__(self, index):
        # returns a (sequence, label) pair 
        return self.X_seq[index], self.y_seq[index]


def get_qkd_dataloaders(csv_path='data/qkd_simulated_telemetry.csv', 
                        window_size=50, 
                        batch_size=64, 
                        train_split=0.8):

    # load telemetry CSV
    df = pd.read_csv(csv_path)
    
    # split into 80% training 20% test and made sure to split sequentially
    # to keep temporal order
    split_idx = int(len(df) * train_split)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    # creates training dataset
    train_dataset = QKDSequenceDataset(
        df=train_df, 
        window_size=window_size, 
        is_train=True
    )
    
    #creates test dataset 
    test_dataset = QKDSequenceDataset(
        df=test_df, 
        window_size=window_size, 
        scaler=train_dataset.scaler, 
        is_train=False
    )
    
    # create the Pytorch dataloaders, which batch the data and move it to the models
    # shuffle = true to prevent model from learning exact sequence from the training set
    # and false for test set to keep chronological order for evaluation 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"DataLoaders successfully created!")
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    print(f"Input tensor shape per batch: ({batch_size}, {window_size}, 3)")

    # return dataloaders and scaler
    return train_loader, test_loader, train_dataset.scaler



if __name__ == '__main__':
    train_loader, test_loader, scaler = get_qkd_dataloaders()
    for X_batch, y_batch in train_loader:
        print("\nBatch Verification:")
        print(f"X_batch shape: {X_batch.shape}")  
        print(f"y_batch shape: {y_batch.shape}")  
        break