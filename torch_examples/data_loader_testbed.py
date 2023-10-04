import os
import pandas as pd

from sklearn.datasets import make_regression

import torch
from torch.utils.data import Dataset, DataLoader


NUM_ROWS = 500_010
NUM_FEATURES = 100
EPOCHS = 2

# Generate synthetic regression data with 100 features and 1000 rows
X, y = make_regression(n_samples=NUM_ROWS, n_features=NUM_FEATURES, random_state=42)

df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(NUM_FEATURES)])
df["target"] = y
print(df.head())

class RegressionDataset(Dataset):
    def __init__(self, df):
        # print the pid of the process for this method
        print(f"pid: {os.getpid()} - init called")
        self.X = torch.tensor(df.drop("target", axis=1).values, dtype=torch.float32)
        self.y = torch.tensor(df["target"].values, dtype=torch.float32)
         
    def __len__(self):
        # print the pid of the process for this method
        # print(f"pid: {os.getpid()} - len called")
        return len(self.X)
    
    def __getitem__(self, idx):
        # print the pid of the process for this method
        # print(f"pid: {os.getpid()} - getitem called, idx: {idx}")
        return self.X[idx], self.y[idx]
    
# Create an instance of the RegressionDataset class
dataset = RegressionDataset(df)

# Create a DataLoader object for the dataset
dataloader = DataLoader(dataset, batch_size=256, num_workers=1, shuffle=True)    

# cycle through the dataloader
for epoch in range(EPOCHS):
    # print pid of this process
    print(f"pid: {os.getpid()} epoch: {epoch}")
    for i, data in enumerate(dataloader, 0):
        # Get the inputs and labels from the dataloader
        inputs, labels = data
        if i % 100 == 0:
            print(f"i {i} inputs shape: {inputs.shape}, labels shape: {labels.shape}")
        

