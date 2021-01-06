import torch
import torchvision
import torchvision.transforms as transform
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time
from model import efficientNet
from Solver import *



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

batch_size = 80
train_dataset = torchvision.datasets.CIFAR10("/home/fred/datasets/",train=True, transform=transformer,download=True)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size= batch_size,shuffle=True,num_workers=2,pin_memory=True)

dev_dataset = torchvision.datasets.CIFAR10("/home/fred/datasets/",train=False, transform=transformer,download=True)
dev_loader = torch.utils.data.DataLoader(dev_dataset,batch_size= batch_size,shuffle=True,num_workers=2,pin_memory=True)


small_train_data = torch.utils.data.Subset(train_dataset, torch.arange(0, 2000))
small_train_loader = torch.utils.data.DataLoader(small_train_data,batch_size=batch_size, num_workers=2,pin_memory=True)

small_dev_data = torch.utils.data.Subset(dev_dataset, torch.arange(0, 100))
small_dev_loader = torch.utils.data.DataLoader(small_dev_data,batch_size=batch_size, num_workers=2,pin_memory=True)


fixrandomseed()
to_float_cuda = {"dtype": torch.float16, "device":"cuda"}

fi = 0
lr = 8e-3
epoch = 2
model = efficientNet(fi=fi, num_classes=10)
model = model.to(**to_float_cuda)
hparam = {'epoch':epoch, 'lr':lr, 'decay':'cosine', 'optim':'SGD nestrov momentum','bsize':batch_size,'fi':fi}
optimizer = torch.optim.SGD(model.parameters(),lr = lr, momentum=0.9,nesterov=True)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epoch)

solver = ClassifierSolver(model, train_loader, dev_loader, optimizer, lr_scheduler)
solver.train(epoch= epoch, verbose=False, hparam= None)
solver.plot()

# model = efficientNet(fi=4, num_classes=10)
# model = model.to(**to_float_cuda)
# optimizer = torch.optim.SGD(model.parameters(),lr = 8e-3, momentum=0.9,nesterov=True)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100)
# load_path = '/home/fred/Python/Mon Jan  4 09:11:12 2021_epoch_94_val_0.9070.tar'
# solver = Solver.load_check_point(load_path, model, train_loader, dev_loader, optimizer,lr_scheduler)
# # solver.train(lr= 1e-3, epoch= 1, verbose=True, checkpoint_name=None)
# solver.plot()


# fn = efficientNet
# args = {'fi':0, 'num_classes':10}
# Sampler(fn, args, small_train_loader, small_dev_loader, num_model=20, epoch=2, lr_lowbound=-5, lr_highbound=-2)
