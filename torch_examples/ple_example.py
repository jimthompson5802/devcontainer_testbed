import sys

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


# Sample data (2D array with 5 columns)
# data = np.random.randn(10, 2)
data = np.array([[0.25, ], [0.1, ], [0.55, ], [0.99,]])
print(f"Original data:\n{data}")

# Define the number of bins
num_bins = 4

# Initialize an empty list to store the bin boundaries for each column
bin_boundaries = []

# Loop through each column and compute the bin boundaries
for i in range(data.shape[1]):  # Iterate over columns
    column_data = data[:, i]  # Extract the current column
    column_min = np.min(column_data) - 1e-8 # Compute the minimum value for the current column
    column_max =  np.max(column_data) + 1e-8  # Compute the maximum value for the current column
    boundaries = np.linspace(column_min, column_max, num_bins + 1) # Compute the bin boundaries for the current column
    bin_boundaries.append(boundaries)  # Add the bin boundaries to the list

print(f"\nBin boundaries:\n{bin_boundaries}")
bin_boundaries = np.vstack(bin_boundaries)  # Convert the list to a 2D array

# Print the bin boundaries for each column
for i, boundaries in enumerate(bin_boundaries):
    print(f"Column {i+1} bin boundaries: {boundaries}")


# Loop through each column and perform piecewise linear encoding
encode_data_list = []
idxs_list = []
encoded_value_list = []
for i in range(data.shape[1]):  # Iterate over columns
    column_data = data[:, i]  # Extract the current column
    column_bin_boundaries = np.array(bin_boundaries[i])  # Get the bin boundaries for the current column

    # Initialize a matrix of all ones to store the encoded data
    encoded_data = np.ones([data.shape[0], num_bins])
    print(f"encoded_data0:\n{encoded_data}")
    
    # Use np.digitize to find the bin indices for each data point
    bin_indices = np.digitize(column_data, column_bin_boundaries) - 1
    print(f"column_bin_boundaries:\n{column_bin_boundaries}")
    print(f"zip(columndata, bin_indices):\n{list(zip(column_data, bin_indices))}")
    
    # Calculate the bin widths based on the bin boundaries
    bin_widths = np.diff(column_bin_boundaries)
    print(f"\nbin_widths:\n{bin_widths}")
    
    
    # Calculate the encoded value of each data point within the selected bin
    encoded_value = (column_data - column_bin_boundaries[bin_indices]) / bin_widths[bin_indices]
    
    # Create a mask to store the encoded value in the corresponding column of encoded_data
    mask = np.zeros_like(encoded_data, dtype=bool)
    mask[np.arange(encoded_data.shape[0]), bin_indices] = True

    # Store the encoded value in the corresponding column of encoded_data
    encoded_data[mask] = encoded_value

    # Create mask to set all values after the column-specific bin index to 0
    mask = np.tile(np.arange(encoded_data.shape[1]), (encoded_data.shape[0], 1))
    mask = mask > bin_indices.reshape(-1, 1)
    encoded_data[mask] = 0


    encode_data_list.append(encoded_data)
    idxs_list.append(bin_indices)
    encoded_value_list.append(encoded_value)

# encoded_data now contains the piecewise linear encoding for each column
encoded_data1 = np.hstack(encode_data_list)
idxs = np.vstack(idxs_list).T
print(f"Encoded data: {encoded_data.shape}\n{encoded_data}")
print(f"idxs: {idxs.shape}\n{idxs}")



class MyTransformer(BaseEstimator, TransformerMixin):
    EPSILON = 1e-8

    def __init__(self, num_bins=4):
        self.num_bins = num_bins    # Store the number of bins

    def fit(self, X, y=None):
        # Fit the transformer to the data
 
        # Initialize an empty list to store the bin boundaries for each column
        bin_boundaries = []

        # Loop through each column and compute the bin boundaries
        for i in range(X.shape[1]):  # Iterate over columns
            column_data = X[:, i]  # Extract the current column
            column_min = np.min(column_data) - self.EPSILON # Compute the minimum value for the current column
            column_max = np.max(column_data) + self.EPSILON  # Compute the maximum value for the current column
            boundaries = np.linspace(column_min, column_max, self.num_bins + 1) # Compute the bin boundaries for the current column
            bin_boundaries.append(boundaries)  # Add the bin boundaries to the list

        # save the bin boundaries for each column, each column in the following matrix is the bin boundaries for the 
        # corresponding column in the data
        self.bin_boundaries = np.vstack(bin_boundaries).T  # Convert the list to a matrix

        return self
    
    def transform(self, X):
        # Transform the data using the fitted transformer
        # Loop through each column and perform piecewise linear encoding
        encode_data_list = []
        for i in range(X.shape[1]):  # Iterate over columns
            column_data = X[:, i]  # Extract the current column
            column_bin_boundaries = np.array(bin_boundaries[i])  # Get the bin boundaries for the current column

            # Initialize a matrix of all ones to store the encoded data
            encoded_data = np.ones([data.shape[0], num_bins])
            
            # Use np.digitize to find the bin indices for each data point
            bin_indices = np.digitize(column_data, column_bin_boundaries) - 1
            
            # Calculate the bin widths based on the bin boundaries
            bin_widths = np.diff(column_bin_boundaries)
            
            # Calculate the encoded value of each data point within the selected bin
            encoded_value = (column_data - column_bin_boundaries[bin_indices]) / bin_widths[bin_indices]
            
            # Create a mask to store the encoded value in the corresponding column of encoded_data
            mask = np.zeros_like(encoded_data, dtype=bool)
            mask[np.arange(encoded_data.shape[0]), bin_indices] = True

            # Store the encoded value in the corresponding column of encoded_data
            encoded_data[mask] = encoded_value

            # Create mask to set all values after the column-specific bin index to 0
            mask = np.tile(np.arange(encoded_data.shape[1]), (encoded_data.shape[0], 1))
            mask = mask > bin_indices.reshape(-1, 1)
            encoded_data[mask] = 0

            encode_data_list.append(encoded_data)

        # stach the encoded data encoded_data into a matrix that contains the piecewise linear encoding for each column
        return np.hstack(encode_data_list)
    

# Create an instance of the transformer
transformer = MyTransformer(num_bins=4)

# Fit the transformer to the data
transformer.fit(data)

# Transform the data using the fitted transformer
encoded_data2 = transformer.transform(data)
print(f"sklearn Encoded data: {encoded_data.shape}\n{encoded_data}")

print(f"encoded_data1 == encoded_data2: {np.allclose(encoded_data1, encoded_data2)}")