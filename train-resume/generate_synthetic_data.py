import os

import pandas as pd

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# constants
N_SAMPLES = 10_000
N_FEATURES = 100


# create synthetic regression dataset
X, y = make_regression(n_samples=N_SAMPLES, n_features=N_FEATURES, noise=0.1, random_state=1)

# convert to pandas dataframe
df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(N_FEATURES)])
df["y"] = y


# print size of dataframe
print(f"size of dataframe: {df.memory_usage(deep=True).sum() / (1024*1024)} MB")
print(df.head())

train_df, test_df = train_test_split(df, test_size=0.2, random_state=1)

# save dataframe to paraquet
os.makedirs("./data", exist_ok=True)
train_df.to_parquet(os.path.join("./data", "train_data.parquet"))
test_df.to_parquet(os.path.join("./data", "test_data.parquet"))
