import torch
import torchvision
import torchvision.transforms as transform
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time
import pickle
from model import efficientNet
from Solver import *



transformer = transform.Compose([
                                transform.Resize(256),
                                transform.CenterCrop(224),
                                transform.ToTensor(),
                                transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

batch_size = 70
train_dataset = torchvision.datasets.CIFAR10("/home/fred/datasets/",train=True, transform=transformer,download=True)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size= batch_size,shuffle=True,num_workers=2,pin_memory=True)

dev_dataset = torchvision.datasets.CIFAR10("/home/fred/datasets/",train=False, transform=transformer,download=True)
dev_loader = torch.utils.data.DataLoader(dev_dataset,batch_size= batch_size,shuffle=True,num_workers=2,pin_memory=True)


small_train_data = torch.utils.data.Subset(train_dataset, torch.arange(0, 200))
small_train_loader = torch.utils.data.DataLoader(small_train_data,batch_size=batch_size, num_workers=2,pin_memory=True)

small_dev_data = torch.utils.data.Subset(dev_dataset, torch.arange(0, 100))
small_dev_loader = torch.utils.data.DataLoader(small_dev_data,batch_size=batch_size*2, num_workers=2,pin_memory=True)


fixrandomseed()

model = efficientNet(fi=0, num_classes=10)
solver = Solver(model, train_loader, dev_loader, print_every_iter= 200, check_every_epoch= 2)

solver.train(lr= 1e-4, epoch= 30, verbose=False, checkpoint_name='test2')
solver.plot()

# solver = Solver.load_checkpoint('test20+_epoch_10.pt', train_loader, dev_loader)
# solver.train(lr= 5e-5, epoch= 10, verbose=True, checkpoint_name='test20+')
# solver.plot()


# fn = efficientNet
# args = {'fi':0, 'num_classes':10}
# Sampler(fn, args, small_train_loader, small_dev_loader, num_model=20, epoch=2, lr_lowbound=-5, lr_highbound=-2)
