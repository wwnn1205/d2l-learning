import torch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = torch.tensor([[2, 1000], [3, 2000], [2, 500], [1, 800], [4, 3000]], dtype=torch.float, device=device)
print(inputs.shape)
outputs = torch.tensor([12,4000],device=device)/inputs
print(outputs)