import numpy as np

# Create a sample 2D array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10 ,11, 12]])
print(f"arr:\n{arr}")

# Create an array of column indices to compare against
col_indices = np.array([1, 0, 2, 0])

# Create a mask that is True for all columns greater than the row-specific column index
mask = np.zeros_like(arr, dtype=bool)
# for i in range(arr.shape[0]):
#     mask[i, col_indices[i]+1:] = True
# mask[np.arange(arr.shape[0]), col_indices+1:] = True

# Create a 4x3 array where the values are the column index
mask2 = np.tile(np.arange(3), (4, 1))

# Print the array
print(mask2)
print(col_indices.reshape(-1, 1))
mask2 = mask2 > col_indices.reshape(-1, 1)
arr[mask2] = 0

# Print the array
print(f"after arr:\n{arr}")