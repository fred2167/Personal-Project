import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import random
import pickle
import time
import math


def fixrandomseed(seed=0):
  '''
  Fix random seed to get deterministic outputs
  '''
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  # random.seed(seed)

def Sampler(model_fn, model_args, train_loader, val_loader, num_model, epoch, lr_lowbound=-5, lr_highbound=0):
  '''
  Use for hyperparameter tuning. Quickly sample learing rates
  Inputs:
    - model_fn: model function
    - model_args: model arguments
    - num_model: number of models for sampling

  '''

  for i in range(num_model):
    model = model_fn(**model_args)
    lr = 10 ** random.uniform(lr_lowbound, lr_highbound)

    print(f'({i+1}/{num_model})lr: {lr:.4e}', end=' ')
    solver = Solver(model, train_loader, val_loader, fp16= False)
    solver.train(epoch, lr)




class Solver(object):
  '''
  Default/ Hard-coded Behavior:
    Using NVDIA GPU, Cuda, Cudnn
    loss function = Cross Entropy Loss


  '''

  def __init__(self, model, train_loader, val_loader, optimizer, lr_scheduler= None, print_every_iter=200, check_every_epoch=2, FP16 = True, random_seed=0):
    '''
    Inputs:
      - optimizer:          Standard Pytorch optimizer. Will ignore preset learning rate of the optimizer and set new learning rate at *train*
      - lr_scheduler:       Standard Pytorch lr scheduler
      - print_every_iter:   print *loss* and *accumilated time* every number of iters, if verbose is also true
      - check_every_epoch:  checkpoint for every number of epoch
      - FP16:               Both model and data are using torch.float16 to save GPU memory, otherwise using torch.float32
      - random_seed:        random seed that make output deterministic. See detail in fixrandomseed function
      
    '''
    self.to_float_cuda = {"dtype": torch.float16 if FP16 else torch.float32, "device":"cuda"}
    self.model = model.to(**self.to_float_cuda)
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.print_every_iter = print_every_iter
    self.check_every_epoch = check_every_epoch

    # Book keeping variables
    self.stats = {}
    stats_names = ['epoch_loss', 'avg_loss', 'train_acc', 'val_acc','ratio']
    for name in stats_names:
      self.stats[name] = []

    self.stats['print_every_iter'] = print_every_iter
    self.stats['check_every_epoch'] = check_every_epoch


    torch.cuda.synchronize()
    torch.backends.cudnn.benchmark = True
    fixrandomseed(random_seed)

    self.optimizer = optimizer
    self.lr_scheduler = lr_scheduler
    self.loss_fn = torch.nn.CrossEntropyLoss()
  

  def train(self, epoch, verbose=False, hparam= None):
    '''Using  tensorboard to log hyperparameters,  during checkpoint
    Inputs:
      - verbose:            print initial loss for sanity check, detail loss during training, otherwise, only print stats in checkpoint
      - hparam:             dictionary of hyperparameters. Will save avg epoch loss, train and validation accuracy to tensorboard.
                            Will save model and optimizer state dict during checkpoint
    '''


    self.epoch = epoch
    self.verbose = verbose
    self.model_start_time = time.ctime()
  
    if hparam:
      writer = SummaryWriter('runs/'+self.model_start_time)

    checkpoint_cycle_flag = True
    num_batch = len(self.train_loader)
    self.model.train()
    for i in range(1, epoch+1):

      total_loss = 0
      iter_loss_history = []
      Y_pred_all = []
      Y_tr_all = []
      start_time = time.time()
      if checkpoint_cycle_flag:
        checkpoint_cycle_flag = False
        checkpoint_start_time = time.time()


      for j, data in enumerate(self.train_loader):
        
        Xtr, Ytr = data
        Xtr, Ytr = Xtr.to(**self.to_float_cuda), Ytr.cuda()

        # alternative of model.zero_grad(). This way is more efficient
        for param in self.model.parameters():
          param.grad = None
        
        y_pred = self.model(Xtr)
        loss = self.loss_fn(y_pred,Ytr)

        total_loss += loss.item()

        loss.backward()
        self.optimizer.step()


        # print training loss per number of iterations
        if verbose and j % self.print_every_iter == 0:
          print(f"Iteration {j}/{num_batch}, loss = {(total_loss / (j+1)):.5f}, took {(time.time() - start_time):.2f} seconds")
        
        # Iter Book keeping
        Y_pred_all.append(y_pred)
        Y_tr_all.append(Ytr)
        iter_loss_history.append(loss.item())

    
      avg_loss = total_loss / num_batch

      if verbose:
        print(f"Epoch: {i}/{epoch}, loss = {avg_loss:.5f}, took {time.time() - start_time:.2f} seconds")

      # Epoch Book keeping
      self.stats['epoch_loss'].append(iter_loss_history)
      self.stats['avg_loss'].append(avg_loss)
        

      
      if i % self.check_every_epoch == 0 or i == epoch or i == 1:
        checkpoint_cycle_flag = True
        epoch_lr = self.optimizer.param_groups[0]['lr']

        # check train accuracy by using saved results during forward pass
        Y_pred_all = torch.argmax(torch.cat(Y_pred_all),dim=1)
        Y_tr_all = torch.cat(Y_tr_all)
        train_accuracy = (Y_pred_all == Y_tr_all).float().mean()

        # check val accuracy
        val_accuracy = self._check_accuracy(self.val_loader)

        # check update ratio
        ratio = self._check_update_ratio(epoch_lr)
        
        print(f'Epoch: {i}/{epoch}, Loss: {avg_loss:.4f} train acc: {train_accuracy:.4f}, val acc: {val_accuracy:.4f}, update ratio: {ratio:.2e}, took {(time.time()-checkpoint_start_time):.2f} seconds, lr: {epoch_lr:.4e}')

        # Checkpoint Book keeping
        self.stats['train_acc'].append(train_accuracy)
        self.stats['val_acc'].append(val_accuracy)
        self.stats['ratio'].append(ratio)
        
        if hparam:
          writer.add_scalar('Epoch loss', avg_loss, i)
          writer.add_scalars('accuracy', {'train': train_accuracy,'val':val_accuracy}, i)
          self._save_checkpoint(epoch=i)

      
      
      # decay learning rate after complete one epoch
      if self.lr_scheduler is not None:
        self.lr_scheduler.step()
    
    # end of training book keeping
    if hparam:
      metrics = {'accuracy':self.stats['val_acc'][-1], 'loss':self.stats['avg_loss'][-1]}
      writer.add_hparams(hparam, metrics, run_name=self.model_start_time)
      writer.flush()
      writer.close()
  

  @torch.no_grad()
  def _check_update_ratio(self, lr):
    '''
    Check the ratio between weights and its graidents.
    Ideal ratio is around 1e-3
    '''
    param_norms = 0
    update_norms = 0
    for param in self.model.parameters():
        param_norms += torch.linalg.norm(param)
        update_norms += torch.linalg.norm(param.grad)
        break # only check the first layer weights??

    return (lr * update_norms / param_norms).item()

  @torch.no_grad()
  def _check_accuracy(self, data_loader, num_sample=None):
    '''
    if num_sample is provided, will sub sample data loader to the designated amount of samples and double the batch size for efficency.
    (Can double the batch size during evaluation since no grads are needed)

    Inputs:
      - data_loader:    Pytorch dataloader object
      - num_sample:     number of samples that are *randomly* picked in the dataloader
    Outputs:
      - acc:            float, accuracy of the dataloader
    
    TODO:
    - Want randomly pick data to calculate accuracy? is making it more deterministic better?
    '''
    self.model.eval()

    if num_sample is not None:
      sub_dataset = torch.utils.data.Subset(data_loader.dataset, torch.randint(high=len(data_loader.dataset),size=(num_sample,)))
      sub_data_loader = torch.utils.data.DataLoader(sub_dataset,data_loader.batch_size*2, num_workers=2,pin_memory=True,)
    else:
      sub_data_loader = data_loader

    Ypred = []
    Yall = []
    for data in sub_data_loader:
      X, Y = data
      X, Y = X.to(**self.to_float_cuda), Y.cuda()

      scores = self.model(X)
      Ypred.append(torch.argmax(scores,dim=1))
      Yall.append(Y)
    
    Ypred = torch.cat(Ypred)
    Yall = torch.cat(Yall)
    acc = (Ypred == Yall).float().mean()

    self.model.train()
    return acc.item()

 
  def _save_checkpoint(self, epoch):

    checkpoint = {
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'epoch':epoch,
      'stats':self.stats
    }
    val_acc = self.stats['val_acc'][-1]
    PATH = f'runs/{self.model_start_time}/{self.model_start_time}_epoch_{epoch}_val_{val_acc:.4f}.tar'
    torch.save(checkpoint, PATH)
    if self.verbose:
      print(f'Saving checkpoint to "{PATH}"')

    

  @staticmethod
  def load_check_point(PATH, model, train_loader, val_loader, optimizer, lr_scheduler= None):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    solver = Solver(model, train_loader, val_loader, optimizer, lr_scheduler)
    solver.stats = checkpoint['stats']
    solver.check_every_epoch = checkpoint['stats']['check_every_epoch']
    solver.print_every_iter = checkpoint['stats']['print_every_iter']
    print(f'load successfully!! check_every_epoch: {solver.check_every_epoch}, print_every_iter: {solver.print_every_iter}')
    return solver

  def plot(self):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.plot(self.stats['epoch_loss'],'oc')
    plt.plot(self.stats['avg_loss'],'-b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(132)
    plt.plot(self.stats['train_acc'],'-o', label='train')
    plt.plot(self.stats['val_acc'],'-o', label='validation')
    plt.xlabel(f'Every {self.check_every_epoch} Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(133)
    plt.plot(self.stats['ratio'],'bo', label='ratio')
    plt.yscale('log')
    plt.yticks([1e-2,1e-3,1e-4,1e-5,1e-6],['1e-2','1e-3','1e-4','1e-5','1e-6'])
    plt.xlabel(f'Every {self.check_every_epoch} Epoch')
    plt.legend()
    plt.show()
