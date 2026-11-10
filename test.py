import numpy as np

data = np.load("data/processed/pie_dataset.npz")

print("Stored arrays:", data.files)

for name in data.files:
    print(name, data[name].shape, data[name].dtype)

data.close()