from preprocess_dataset import load_dataset

data, genes, cells = load_dataset()
print("Data shape:", data.shape)
