import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch

class QKDSequenceDataset(Dataset):
    def __init__(self, df, window_size=50, scaler=None, is_train=True):
        self.window_size = window_size
        
        # 1. Use the DataFrame passed from get_qkd_dataloaders
        X = df[["qber", "jitter", "dark_counts"]].to_numpy()
        y = df["attack_label"].to_numpy()

        # 2. Fit and store scaler on instance if training
        if is_train:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else: 
            self.scaler = scaler
            X_scaled = self.scaler.transform(X)
            
        self.X_seq = []
        self.y_seq = []
        for i in range(0, len(X_scaled) - window_size):
            self.X_seq.append(X_scaled[i : i + window_size])
            self.y_seq.append(y[i + window_size - 1])
            
        # 3. Convert self.X_seq and self.y_seq to PyTorch tensors
        self.X_seq = torch.tensor(np.array(self.X_seq), dtype=torch.float32)
        self.y_seq = torch.tensor(np.array(self.y_seq), dtype=torch.float32)
    
    def __len__(self):
        return len(self.X_seq)
    
    def __getitem__(self, index):
        return self.X_seq[index], self.y_seq[index]


def get_qkd_dataloaders(csv_path='data/qkd_simulated_telemetry.csv', 
                        window_size=50, 
                        batch_size=64, 
                        train_split=0.8):
    """
    Loads CSV, performs a temporal split, scales data, and returns PyTorch DataLoaders.
    """
    # 1. Load telemetry CSV
    df = pd.read_csv(csv_path)
    
    # 2. Chronological Split (e.g., 80% Train, 20% Test)
    split_idx = int(len(df) * train_split)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    # 3. Create Train Dataset & Fit Scaler
    train_dataset = QKDSequenceDataset(
        df=train_df, 
        window_size=window_size, 
        is_train=True
    )
    
    # 4. Create Test Dataset using the fitted Scaler from Train
    test_dataset = QKDSequenceDataset(
        df=test_df, 
        window_size=window_size, 
        scaler=train_dataset.scaler, 
        is_train=False
    )
    
    # 5. Build PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"DataLoaders successfully created!")
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    print(f"Input tensor shape per batch: ({batch_size}, {window_size}, 3)")
    
    return train_loader, test_loader, train_dataset.scaler


# --- Quick Test Block ---
if __name__ == '__main__':
    train_loader, test_loader, scaler = get_qkd_dataloaders()
    for X_batch, y_batch in train_loader:
        print("\nBatch Verification:")
        print(f"X_batch shape: {X_batch.shape}")  # Should be [64, 50, 3]
        print(f"y_batch shape: {y_batch.shape}")  # Should be [64]
        break