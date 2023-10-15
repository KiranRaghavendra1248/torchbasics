# Imports
import torch


# Parameters
device = "cuda" if torch.cuda.is_available() else "cpu"



# Code
tensor = torch.tensor([[1,2,3],[4,5,6]],device=device,dtype=torch.float64)
print(tensor)
print(tensor.shape)

tensor = torch.empty(size=(3,3))
print("Empty\n",tensor)

tensor = torch.ones(size=(3,3))
print("Ones\n",tensor)

tensor = torch.zeros(size=(3,3))
print("Zeros\n",tensor)

tensor = torch.rand(size=(3,3))
print("Random\n",tensor)

tensor = torch.eye(5,5)
print("Identity\n",tensor)

tensor = torch.linspace(start=0.1,end=1,steps=5)
print("Linear distributed\n",tensor)

tensor = torch.empty(size=(1,5)).normal_(mean=0,std=1)
print("Normal distribution\n",tensor)

tensor = torch.empty(size=(1,5)).uniform_(0,1)
print("Uniform distribution\n",tensor)