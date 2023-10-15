# Imports
import torch

# Parametes


x = torch.tensor([1,2,4])
y = torch.tensor([3,5,8])
print(x,y)

# Element wise Addition
z = torch.add(x,y)
print(z)
z = x+y
print(z)

# Element wise Subtraction
z = x-y
print(z)

# Element wise Division
z = torch.true_divide(x,y)
print(z)
z = torch.true_divide(x,2)
print(z)

# Inplace operation(w/o creating a copy)
x.add_(y)  # functions followed by _ indicates inplace operations
print(x)
x+=y   # note : x = x+y is not inplace, it creates a copy
print(x)

# Element wise Exponentiation
z = x.pow(2)
print(z)
z = x**(2)
print(z)

# Simple Comparison
z = x>8
print(z)

# Matrix Multiplication
x1 = torch.rand(size=(2,5))
x2 = torch.rand(size=(5,3))
x3 = torch.mm(x1,x2) # 2x3
print(x3)
x3 = x1.mm(x2) # 2x3
print(x3)

# Matrix Exponentiation (Multiplying a matrix with itself N times)
x = torch.rand(size=(5,5))
print(x.matrix_power(3))

# Element wise multiplication of 2 matrices
x1 = torch.rand(size=(3,4))
x2 = torch.rand(size=(3,4))
z = x1*x2
print(z)

# Dot product
x1 = torch.rand(4)
x2 = torch.rand(4)
z = torch.dot(x1,x2)
print(z)

# Batch multiplication
b = 64
m,n,p = 5,8,10
x1 = torch.rand(size=(b,m,n))
x2 = torch.rand(size=(b,n,p))
out_bmm = torch.bmm(x1,x2)
print(out_bmm)

# Broadcasting
x1 = torch.rand(size=(5,5))
x2 = torch.rand(size=(1,5))

x3 = x1-x2
print(x3)

# Tensor Operations
x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(torch.sum(x,dim=0)) # Add across dimension 0 or Reduce dimension 0 (Results in addition across rows)
print(torch.sum(x,dim=1)) # Add across dimension 1 or Reduce dimension 1 (Results in addition across cols)
