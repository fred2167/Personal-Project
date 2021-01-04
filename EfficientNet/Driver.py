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

model = efficientNet(fi=0, num_classes=10)
model = model.to(**to_float_cuda)
optimizer = torch.optim.RMSprop(model.parameters(), momentum=0.9, eps=1e-4,weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

solver = Solver(model, small_train_loader, small_dev_loader, optimizer, lr_scheduler, print_every_iter= 200, check_every_epoch= 2)
solver.train(lr= 1e-3, epoch= 5, verbose=False, checkpoint_name=None)
solver.plot()

# model_fn = efficientNet
# model_args = {'fi':0, 'num_classes':10}
# load_path = '/home/fred/Python/Sat Jan  2 19:55:37 2021_epoch_20_val_0.8895.tar'
# solver = Solver.load_check_point(load_path, model_fn, model_args, train_loader, dev_loader)
# solver.train(lr= 4e-4, epoch= 20, verbose=False, checkpoint_name=time.ctime())
# solver.plot()


# fn = efficientNet
# args = {'fi':0, 'num_classes':10}
# Sampler(fn, args, small_train_loader, small_dev_loader, num_model=20, epoch=2, lr_lowbound=-5, lr_highbound=-2)
