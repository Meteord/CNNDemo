import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = nn.Conv2d(3,3,5,1,2)
        self.conv0_bn = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 6, 5) #3 in channels, 6 outchannels 5 kernelsize
        self.conv1_bn = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6,3,1,1)
        self.conv2_bn = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6, 16, 3)
        self.conv3_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # x is shape (batchsize,3, 32,32])
    def forward(self, x): 
        x = F.relu(self.conv0_bn(self.conv0(x))) #same conv
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = F.relu(self.conv2_bn(self.conv2(x))) #same conv
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x)))) #batchsize,400
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
