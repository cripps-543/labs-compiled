import torch


# Create two tensors (matrices)
matrix1 = torch.tensor([[1, 2], [3, 4]])
matrix2 = torch.tensor([[5, 6], [7, 8]])


# Perform matrix multiplication
mul = torch.matmul(matrix1, matrix2)
print(mul)