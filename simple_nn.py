## Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

## Create FC NN
class SimpleNN(nn.Module):
    def __init__(self,input_size,classes):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(784,50)
        self.fc2 = nn.Linear(50, classes)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN(784,10)
"""
x = torch.rand(64,784)
y = model.forward(x)
print(y.shape)
"""

## Set device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Running on {0}..".format(device))

## Hyperparams
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

## Load Data
train_dataset = datasets.MNIST(root='dataset/',download=True,train=True,transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dataset = datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

## Create model
model = SimpleNN(input_size=input_size,classes=num_classes).to(device=device)

## Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# Note : CrossEntropyLoss() can take the truth value as class indices or one hot encoded vectors both

## Train
for epoch in range(num_epochs):
    for batch_idx,(data,target) in enumerate(train_loader):
        # get data to cuda
        data = data.to(device=device)
        target = target.to(device=device)

        """
        To keep the assert statement below, we need to include drop_last option in dataloader, as dataset size may not be an integer multiple of batch_size, hence this assert will fail for the last batch
        """
        # assert data.shape == (batch_size,1,28,28) "Data shape mismatch"
        # assert target.shape == (batch_size), "Target shape mismatch"

        data = data.reshape((data.shape[0],-1))

        # forward prop
        target_pred = model(data)
        loss = criterion(target_pred,target)

        # backward prop
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        print("Running epoch {0}. Batch {1}".format(epoch,batch_idx),end='\r')

# Check accuracy
def check_accuracy(loader,model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct,num_samples = 0,0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            x = x.reshape((x.shape[0],-1))

            y_pred = model(x)
            _, predictions = torch.max(y_pred,dim=1)
            num_correct += (predictions==y).sum()
            num_samples += predictions.shape[0]

        print(f'Got {num_correct}/{num_samples} with accuracy {(float(num_correct)/float(num_samples))*100:.2f}')

    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader, model)





