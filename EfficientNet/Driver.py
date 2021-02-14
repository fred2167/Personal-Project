import torch
import torchvision
import torchvision.transforms as transform
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time
from model import efficientNet
from Solver import *
from Optim import AdaBelief



transformer = transform.Compose([
                                transform.Resize(256),
                                transform.CenterCrop(224),
                                # transform.RandomRotation(90),
                                transform.RandomHorizontalFlip(),
                                transform.RandomVerticalFlip(),
                                transform.ToTensor(),
                                transform.RandomErasing(),
                                transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

batch_size = 64
train_dataset = torchvision.datasets.CIFAR10("/home/fred/datasets/",train=True, transform=transformer,download=True)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size= batch_size,shuffle=True,num_workers=2,pin_memory=True)

dev_dataset = torchvision.datasets.CIFAR10("/home/fred/datasets/",train=False, transform=transformer,download=True)
dev_loader = torch.utils.data.DataLoader(dev_dataset,batch_size= batch_size,shuffle=True,num_workers=2,pin_memory=True)


small_train_data = torch.utils.data.Subset(train_dataset, torch.arange(0, 2000))
small_train_loader = torch.utils.data.DataLoader(small_train_data,batch_size=batch_size, num_workers=2,pin_memory=True)

small_dev_data = torch.utils.data.Subset(dev_dataset, torch.arange(0, 500))
small_dev_loader = torch.utils.data.DataLoader(small_dev_data,batch_size=batch_size, num_workers=2,pin_memory=True)


fixrandomseed()
to_float_cuda = {"dtype": torch.float32, "device":"cuda"}

fi = 0
lr = 3e-3
epoch = 10
model = efficientNet(fi=fi, num_classes=10)
model = model.to(**to_float_cuda)

optimizer = torch.optim.SGD(model.parameters(),lr = lr, momentum=0.9,nesterov=True)
# optimizer = torch.optim.Adam(model.parameters(), lr= lr)
# optimizer = AdaBelief(model.parameters(), lr= lr)

# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epoch)
lr_scheduler = None


# hparam = {'epoch':epoch, 'optim':'SGD', 'lr':lr, 'lr decay': 'Cosine', 'bsize':batch_size}
hparam = None


solver = ClassifierSolver(model, train_loader, dev_loader, optimizer, lr_scheduler)
solver.train(epoch, hparam)
solver.plot()


# load_path = '/home/fred/Python/Personal-Project/EfficientNet/runs/Sat Jan 16 20:46:33 2021/Sat Jan 16 20:46:33 2021_epoch_50_val_0.8822.tar'
# solver = ClassifierSolver.load_check_point(load_path, model, train_loader, dev_loader, optimizer, lr)
# solver.train(epoch, hparam)



# fn = efficientNet
# args = {'fi':0, 'num_classes':10}
# Sampler(fn, args, train_loader, dev_loader, num_model=20, epoch=1, lr_lowbound=-5, lr_highbound=-2.5)
