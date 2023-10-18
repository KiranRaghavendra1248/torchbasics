# Imports
import torch
import numpy as np


# Parameters
device = "cuda" if torch.cuda.is_available() else "cpu"



# Initialization and Basic Functions
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

tensor = torch.rand(3)
print("Tensor\n",tensor)
print("I/P 1D -> O/P 2D on main diagonal\n", torch.diag(tensor, 0))

tensor = torch.rand(3)
print("Tensor\n",tensor)
print("I/P 1D -> O/P 2D above main diagonal\n", torch.diag(tensor,1))

tensor = torch.rand(3)
print("Tensor\n",tensor)
print("I/P 1D -> O/P 2D below main diagonal\n", torch.diag(tensor,-1))

tensor = torch.rand(size=(3,3))
print("Tensor\n",tensor)
print("I/P 2D -> O/P 1D - main diagonal\n", torch.diag(tensor,0))

tensor = torch.rand(size=(3,3))
print("Tensor\n",tensor)
print("I/P 2D -> O/P 1D - above main diagonal\n", torch.diag(tensor,1))

tensor = torch.rand(size=(3,3))
print("Tensor\n",tensor)
print("I/P 2D -> O/P 1D - below main diagonal\n", torch.diag(tensor,-1))


# Conversion from one dtype to another
tensor = torch.arange(5)
print(tensor.dtype)
# default = 64 bit for int
print(tensor.bool()) # bool
print(tensor.short()) # int16
print(tensor.long()) # int 64
print(tensor.half()) # float 16
print(tensor.float()) # float32
print(tensor.double()) # float64

# Conversion from numpy to tensor and tensor to numpy
np_array = np.zeros(shape=(3,3))
tensor = torch.from_numpy(np_array)
print("Numpy array\n",np_array)
print("Tensor\n",tensor.shape, tensor)

np_array_back = tensor.numpy()
print(np_array_back)