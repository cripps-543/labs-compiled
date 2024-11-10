import torch


# Create a tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Reshape the tensor
reshaped = torch.reshape(tensor, (3,2))
print(reshaped)