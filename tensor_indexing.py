import torch

## tensor indexing
batch_size = 32
features = 25

x = torch.rand(size=(batch_size,features))
print(x[0].shape)

print(x[:,0].shape)

# Get the 3rd eg, first10 features
print(x[2,:10].shape)

# fancy indexing
x = torch.arange(10,20)
indices = [2,4,6]
print(x[indices])

# pick out specific elements in a matrix
x = torch.rand(size=(3,6))
rows = [1,2]
cols = [4,5]

print(x[rows,cols])

# advanced indexing - pandas kind of a thing, basically selecting elements using a bool array
x = torch.arange(20)
print((x<5) | (x>15))
print(x[(x<5) | (x>15)])

# useful operations
print(torch.where(x>5,x,3*x))
x = torch.rand(size=(3,6,5))
print(x.ndimension()) # prints num of dimensions
print(x.numel()) # count elements

# tensor_reshaping
x = torch.arange(9)

x3x3 = x.view((3,3)) # tensor is stored contiguously in memory - may lead to errors, use .contiguous() incase of error
print(x3x3.shape)

x3x3 = x.reshape((3,3)) # doesnt really matter - it makes a copy - may lead to performance loss
print(x3x3.shape)

# tensor concat
x1 = torch.rand(size=(3,5))
x2 = torch.rand(size=(3,5))

print(torch.cat((x1,x2),dim=0).shape)  # concatenate across dim 0
print(torch.cat((x1,x2),dim=1).shape)  # concatenate across dim 1

# unroll a tensor
batch_size = 64
x = torch.rand(size=(batch_size,5,8))
print(x.reshape(batch_size,-1).shape) # 64x40
# keeps the dimension which you specify and unrolls other dimensions into a big long vector

# change the axis for tensor - eg : 65x5x8 : 65x8x5
print(x.permute(0,2,1).shape)

# add a dimension
x = torch.arange(10)
print(x.unsqueeze(0).shape) # adds dim at pos 0 - makes it 1x10
print(x.unsqueeze(1).shape) # adds dim at pos 1 - makes it 1x10

# remove a dimension
print()

# concatenation 2 batches together of size=16,10
x1 = torch.rand(size=(16,48,48))
x2 = torch.rand(size=(10,48,48))

print(torch.cat((x1,x2),dim=0).shape) # 26x48x48 , note : here dim apart from 0, must be of same size

x1 = torch.rand(size=(2,4))
x2 = torch.rand(size=(2,4))
print(torch.stack((x1,x2), dim=0).shape)
# stack = unsqueeze + cat i.e it creates a new dim and cats it, so the other dim must be matching in size
