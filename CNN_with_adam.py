# Deep Learning with Pytorch
# Module 5: Convolutional Neural Networks (CNN)
# CNN Challenge on CIFAR10 dataset

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
# Step 1: Setup
torch.manual_seed(1)
writer = SummaryWriter()

# Hyper Parameters
EPOCH = 2
BATCH_SIZE = 128
LR = 0.001

train_data = torchvision.datasets.CIFAR10(
    root='./cifar10',
    train=True,
    download=True,
    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True)

test_data = torchvision.datasets.CIFAR10(
    root='./cifar10',
    train=False,
    download=True,
    transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Step 2: Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NASoptim(Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.9, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(NASoptim, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NASoptim, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            #print(weight_decay, momentum, dampening , nesterov)
            #They use pytorch functions most likely to speed up the operations
            #More research is defiently needed to understand how optimisers are implemented in pytorch.

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros(p.data.size()).cuda()
                        buf = buf.mul(momentum).add(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf = buf.mul(momentum).add(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        #This is the gradient update rule that was found in the paper Neural Optimizer search with reinfrocmement learning
                        d_p = torch.mul(torch.exp(d_p.sign().mul(buf.sign())),d_p)
                        #print(d_p)
                        #d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
cnn = CNN()
cnn.cuda()
print(cnn)

# Step 3: Loss Funtion
loss_func = nn.CrossEntropyLoss()

#Step 4: Optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
#optimizer = torch.optim.sgd(cnn.parameters(), lr=LR)
#optimizer = NASoptim(cnn.parameters(), lr=LR)

#Step 5: Training Loop
for epoch in range(EPOCH):  # loop over the dataset multiple times
    for i, (x,y) in enumerate(train_loader):
        #x, y = Variable(x), Variable(y)
        x, y = Variable(x.cuda()), Variable(y.cuda())

        yhat = cnn(x)
        loss = loss_func(yhat, y)    # cross entropy loss
        writer.add_scalar('loss', loss.data[0], i)

        optimizer.zero_grad()            # clear gradients for this training step
        loss.backward()                  # backpropagation, compute gradients
        optimizer.step()                 # apply gradients

        _,y_pred = torch.max(yhat.data, 1)
        total = y.size(0)
        correct = (y_pred == y.data).sum()
        writer.add_scalar('accuracy', (100 * correct / total), i)
        if i % 10 == 0:
            print('Epoch/Step: {}/{}'.format(epoch+1,i),
                '| train loss: %.4f' % loss.data[0],
                '| accuracy: %.2f %%' % (100 * correct / total))

#Step 6: Evaluation
for (x,y) in test_loader:
    yhat = cnn(Variable(x).cuda())
    _, y_pred = torch.max(yhat.data, 1)
    total = y.size(0)
    correct = (y_pred == y.cuda()).sum()
print('Test accuracy: %.2f %%' % (100 * correct / total))
writer.export_scalars_to_json("./all_scalars.json")

writer.close()
