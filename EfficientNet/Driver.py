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
                                transform.RandomRotation(90),
                                transform.RandomHorizontalFlip(),
                                transform.RandomVerticalFlip(),
                                transform.ToTensor(),
                                transform.RandomErasing(),
                                transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

batch_size = 80
train_dataset = torchvision.datasets.CIFAR10("/home/fred/datasets/",train=True, transform=transformer,download=True)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size= batch_size,shuffle=True,num_workers=2,pin_memory=True)

dev_dataset = torchvision.datasets.CIFAR10("/home/fred/datasets/",train=False, transform=transformer,download=True)
dev_loader = torch.utils.data.DataLoader(dev_dataset,batch_size= batch_size,shuffle=True,num_workers=2,pin_memory=True)


small_train_data = torch.utils.data.Subset(train_dataset, torch.arange(0, 200))
small_train_loader = torch.utils.data.DataLoader(small_train_data,batch_size=batch_size, num_workers=2,pin_memory=True)

small_dev_data = torch.utils.data.Subset(dev_dataset, torch.arange(0, 100))
small_dev_loader = torch.utils.data.DataLoader(small_dev_data,batch_size=batch_size, num_workers=2,pin_memory=True)


fixrandomseed()

# model = efficientNet(fi=0, num_classes=10)
# checkpoint_PATH = '/home/fred/Python/Personal-Project/EfficientNet/CheckPoints/test2'
# solver = Solver(model, small_train_loader, small_dev_loader, print_every_iter= 200, check_every_epoch= 2)
# solver.train(lr= 1e-4, epoch= 1, verbose=True, checkpoint_PATH=checkpoint_PATH)
# solver.plot()

model_fn = efficientNet
model_args = {'fi':0, 'num_classes':10}
load_path = '/home/fred/Python/Personal-Project/EfficientNet/CheckPoints/test2_epoch_1.tar'
solver = Solver.load_check_point(load_path, model_fn, model_args, small_train_loader, small_dev_loader)
solver.train(lr= 1e-4, epoch= 1, verbose=True, checkpoint_PATH=None)
solver.plot()


# fn = efficientNet
# args = {'fi':0, 'num_classes':10}
# Sampler(fn, args, small_train_loader, small_dev_loader, num_model=20, epoch=2, lr_lowbound=-5, lr_highbound=-2)
