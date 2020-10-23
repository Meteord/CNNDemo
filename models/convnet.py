import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = nn.Conv2d(3,3,5,1,2)
        self.conv1 = nn.Conv2d(3, 6, 5) #3 in channels, 6 outchannels 5 kernelsize
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6,3,1,1)
        self.conv3 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # x is shape (batchsize,3, 32,32])
    def forward(self, x): 
        x = F.relu(self.conv0(x)) #same conv
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x)) #same conv
        x = self.pool(F.relu(self.conv3(x))) #batchsize,400
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
