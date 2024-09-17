import cuml
from cuml.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time
import numpy as np


# conda activate rapids-24.06

# Load the iris dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
# Define the fraction to increase the size by (e.g., 1.6 means increase by 60%)
fraction = 1

# Accept GPU_N and fraction as command-line arguments
if len(sys.argv) != 3:
    print("Usage: python pureCupy.py <GPU_N> <fraction>")
    sys.exit(1)

GPU_N = int(sys.argv[1])
fraction = float(sys.argv[2])

# Calculate the new number of rows
original_num_rows = X.shape[0]
new_num_rows = int(original_num_rows * fraction)

# Use np.tile to repeat the original array
num_repeats = int(np.ceil(fraction))
tiled_X = np.tile(X, (num_repeats, 1))
tiled_y = np.tile(y, num_repeats)

# Slice to get the desired number of rows
X_resized = tiled_X[:new_num_rows]
y_resized = tiled_y[:new_num_rows]

# Convert data types
# takes circa 2 seconds
# X= X_resized.astype(np.float32)
# y = y_resized.astype(np.int32)

# Convert data types
# takes circa 13.6 seconds
X= X_resized.astype(np.int32)
y = y_resized.astype(np.int32)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
start_time = time.time()
# Initialize the GPU-based KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time with GPU(s): {elapsed_time} ")
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
