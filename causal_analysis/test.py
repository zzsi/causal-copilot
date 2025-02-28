import numpy as np

# Load the file
data = np.load('causal_analysis/real_data/ihdp_npci_1-1000.train.npz')

# See what arrays are stored in the file
print(data.files)
print(data['ymul'].shape)