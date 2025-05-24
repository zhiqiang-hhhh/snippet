# pip install usearch

import numpy as np
import usearch  # Import the package directly

index = usearch.Index(ndim=3)               # Default settings for 3D vectors
vector = np.array([0.2, 0.6, 0.4])  # Can be a matrix for batch operations
index.add(42, vector)               # Add one or many vectors in parallel
matches = index.search(vector, 10)  # Find 10 nearest neighbors

assert matches[0].key == 42
assert matches[0].distance <= 0.001
assert np.allclose(index[42], vector, atol=0.1) # Ensure high tolerance in mixed-precision comparisons

index = usearch.Index(
    ndim=3, # Define the number of dimensions in input vectors
    metric='ip', # Choose 'l2sq', 'ip', 'haversine' or other metric, default = 'cos'
    dtype='bf16', # Store as 'f64', 'f32', 'f16', 'i8', 'b1'..., default = None
    connectivity=16, # Optional: Limit number of neighbors per graph node
    expansion_add=128, # Optional: Control the recall of indexing
    expansion_search=64, # Optional: Control the quality of the search
    multi=False, # Optional: Allow multiple vectors per key, default = False
)

index.add(42, vector)
matches = index.search(vector, 10)
for match in matches:
    print(match.key, match.distance)