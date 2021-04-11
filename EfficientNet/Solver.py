import torch
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import random
import time
import numpy as np
from tqdm import tqdm, trange
import wandb



def fixrandomseed(seed=0):
  '''
  Fix random seed to get deterministic outputs
  '''
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  # random.seed(seed)

def Sampler(model_fn, model_args, train_loader, val_loader, num_model, epoch, lr_lowbound=-5, lr_highbound=0):
  '''
  Use for hyperparameter tuning. Quickly sample learing rates
  Inputs:
    - model_fn: model function
    - model_args: model arguments
    - num_model: number of models for sampling
  '''
  to_float_cuda = {"dtype": torch.float32, "device":"cuda"}

  for i in range(num_model):
    lr = 10 ** random.uniform(lr_lowbound, lr_highbound)

    model = model_fn(**model_args).to(**to_float_cuda)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    
    solver = ClassifierSolver(model, train_loader, val_loader, optimizer,None, epoch)
    solver.train(epoch)
    



class Solver(object):
  '''
  Default/ Hard-coded Behavior:
    Using NVDIA GPU, Cuda, Cudnn
  '''

  def __init__(self, model, train_loader, val_loader, optimizer, lr_scheduler= None, check_every_epoch=1, random_seed=0):
    '''
    Inputs:
      - optimizer:          Standard Pytorch optimizer. Will ignore preset learning rate of the optimizer and set new learning rate at *train*
      - lr_scheduler:       Standard Pytorch lr scheduler
      - check_every_epoch:  checkpoint for every number of epoch
      - :        random seed that make output deterministic. See detail in fixrandomseed function
      
    '''
    self.to_float_cuda = {"dtype": torch.float32, "device":"cuda"}
    self.model = model.to(**self.to_float_cuda)
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.optimizer = optimizer
    self.lr_scheduler = lr_scheduler
    self.scaler = amp.GradScaler()




    # Book keeping variables
    self.config = dict(check_every_epoch=check_every_epoch, 
                       model_start_time = time.ctime(),
                       previous_epoch = 0)


    self.stats = dict(iter_loss = [],
                      avg_loss = [],
                      train_acc = [],
                      val_acc = [],
                      ratio = [])


    torch.cuda.synchronize()
    torch.backends.cudnn.benchmark = True
    fixrandomseed(random_seed)
  
  def train_fn(self, Xtr, Ytr):
    '''
    TODO: elimiate the need to output y_pred, need to test memory footprint
    Function that all sub class need to implement
    Input:
      - Xtr, Ytr:   Training data and labels that are given by dataloader as ONE small batch
    Outputs: (**Order Matters**)
      - loss:       scalar, One iteration loss
      - y_pred:     prediction that that are made by the model. Have to be the same size and shape with Ytr
    '''
    raise NotImplementedError

  def train(self, epoch, hparam= None):
    '''
    Inputs:
      - hparam:             dictionary of hyperparameters. 
                            Save average epoch loss, train and validation accuracy to tensorboard.
                            After half of the training epoch, save model, optimizer ,and scalar state dict, current epoch, stats and config during checkpoint 
    '''

  
    model_start_time = self.config['model_start_time']
    previous_epoch = self.config['previous_epoch']
    check_every_epoch = self.config['check_every_epoch']

    self.config['hparam'] = hparam

    epoch += previous_epoch # for load previous model

    if hparam:
      writer = SummaryWriter('runs/'+model_start_time)
      self.track = wandb.init(config=hparam)
      wandb.watch(self.model)


    
    checkpoint_cycle_flag = True
    num_batch = len(self.train_loader)
    self.model.train()
    for i in range(previous_epoch+1, epoch+1):

      total_loss = 0
      iter_loss_history = []
      Y_pred_all = []
      Y_tr_all = []

      if checkpoint_cycle_flag:
        checkpoint_cycle_flag = False
        checkpoint_start_time = time.time()

      for j, data in zip(s:=trange(num_batch, leave=False),self.train_loader):
        

        Xtr, Ytr = data
        Xtr, Ytr = Xtr.to(**self.to_float_cuda), Ytr.cuda()

        ################################## Future changes ##########################################################

        loss, y_pred = self.train_fn(Xtr, Ytr)

        ############################################################################################################

        total_loss += loss
        
        # Iter Book keeping
        Y_pred_all.append(y_pred)
        Y_tr_all.append(Ytr)
        iter_loss_history.append(loss)

        # update progress bar
        s.set_description(f'Epoch {i}/{epoch} Loss: {loss:.4f} ')

    
      avg_loss = total_loss / num_batch

      # Epoch Book keeping
      self.stats['iter_loss'].append(iter_loss_history)
      self.stats['avg_loss'].append(avg_loss)
        

      # Enter checkpoint block after first and last epoch and specify checkpoint interval
      if i % check_every_epoch == 0 or i == epoch:
        checkpoint_cycle_flag = True
        cur_lr = self.optimizer.param_groups[0]['lr']

        # check train accuracy by using saved results during forward pass to save computation. 
        Y_pred_all = torch.argmax(torch.cat(Y_pred_all),dim=1)
        Y_tr_all = torch.cat(Y_tr_all)
        train_accuracy = (Y_pred_all == Y_tr_all).float().mean()

        # check val accuracy
        val_accuracy, val_loss = self._check_accuracy(self.val_loader)
 
        # check update ratio
        ratio = self._check_update_ratio(cur_lr)
        
        print(f'Epoch: {i}/{epoch}, train loss: {avg_loss:.4f}, val loss: {val_loss:.4f}, train acc: {train_accuracy:.4f}, val acc: {val_accuracy:.4f},lr: {cur_lr:.4e}, update ratio: {ratio:.2e}, took {(time.time()-checkpoint_start_time):.2f} seconds')

        # Checkpoint Book keeping
        self.stats['train_acc'].append(train_accuracy)
        self.stats['val_acc'].append(val_accuracy)
        self.stats['ratio'].append(ratio)

        
        
        if hparam:
          writer.add_scalar('Epoch loss', avg_loss, i)
          writer.add_scalars('accuracy', {'train': train_accuracy,'val':val_accuracy}, i)
          wandb.log({"train loss": avg_loss, "val loss": val_loss,"train acc": train_accuracy, "val acc": val_accuracy})

          # only save model checkpoint after half of the training process
          if i > epoch//2:
            self._save_checkpoint(epoch=i)

      
      
      # decay learning rate after complete one epoch
      if self.lr_scheduler is not None:
        self.lr_scheduler.step()
    

    # end of training book keeping
    if hparam:
      metrics = {'Val Accuracy':self.stats['val_acc'][-1], 'Loss':self.stats['avg_loss'][-1]}
      writer.add_hparams(hparam, metrics, run_name='hparam')
      writer.add_figure(self.config['model_start_time']+str(hparam), self.plot())
      writer.flush()
      writer.close()
      
  

  @torch.no_grad()
  def _check_update_ratio(self, lr):
    '''
    Check the ratio between weights and its graidents.
    Ideal ratio is around 1e-3

    param.grad is the *raw* gradient
    '''
    param_norms = 0
    grad_norms = 0
    for param in self.model.parameters():
        param_norms += torch.linalg.norm(param)
        grad_norms += torch.linalg.norm(param.grad)
        # break # only check the first layer weights??

    return (lr * grad_norms / param_norms).item()

  @torch.no_grad()
  def _check_accuracy(self, data_loader, num_sample=None):
    '''
    TODO:
    - Want randomly pick data to calculate accuracy? is making it more deterministic better?
    if num_sample is provided, will sub sample data loader to the designated amount of samples and double the batch size for efficency.
    (Can double the batch size during evaluation since no grads are needed)
    Inputs:
      - data_loader:    Pytorch dataloader object
      - num_sample:     number of samples that are *randomly* picked in the dataloader
    Outputs:
      - acc:            float, accuracy of the dataloader
    
    
    '''
    self.model.eval()

    if num_sample is not None:
      sub_dataset = torch.utils.data.Subset(data_loader.dataset, torch.randint(high=len(data_loader.dataset),size=(num_sample,)))
      sub_data_loader = torch.utils.data.DataLoader(sub_dataset,data_loader.batch_size*2, num_workers=2,pin_memory=True,)
    else:
      sub_data_loader = data_loader

    Ypred = []
    Yall = []
    total_loss = 0
    for data in sub_data_loader:
      X, Y = data
      X, Y = X.to(**self.to_float_cuda), Y.cuda()

      with amp.autocast():
        scores = self.model(X)
        loss = torch.nn.CrossEntropyLoss()(scores, Y)

      total_loss += loss
      Ypred.append(torch.argmax(scores,dim=1))
      Yall.append(Y)
    
    Ypred = torch.cat(Ypred)
    Yall = torch.cat(Yall)
    acc = (Ypred == Yall).float().mean()

    self.model.train()
    return acc.item(), (total_loss / len(sub_data_loader)).item()

 
  def _save_checkpoint(self, epoch):
    self.config['previous_epoch'] = epoch

    checkpoint = {
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'scaler_state_dict':self.scaler.state_dict(),
      'stats':self.stats,
      'config':self.config
    }
    val_acc = self.stats['val_acc'][-1]
    model_start_time = self.config['model_start_time']
    PATH = f'runs/{model_start_time}/{model_start_time}_epoch_{epoch}_val_{val_acc:.4f}.tar'

    torch.save(checkpoint, PATH)

    # weights and biases tracking
    artifact = wandb.Artifact("model", type='model_state')
    artifact.add_file(PATH)
    self.track.log_artifact(artifact)

  @staticmethod
  def _load_check_point(INSTANCE, PATH, model, train_loader, val_loader, optimizer, lr, lr_scheduler= None):
    '''
    Sub-class should overload this method with INSTANCE
    '''
    checkpoint = torch.load(PATH)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    #overide the learning rate in old optimizer
    for p in optimizer.param_groups:
      p['lr'] = lr

    solver = INSTANCE(model, train_loader, val_loader, optimizer, lr_scheduler)
    solver.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    solver.stats = checkpoint['stats']
    solver.config = checkpoint['config']
    

    config = checkpoint['config']
    print(f'load successfully!! {str(config)}')
    return solver

    

  def plot(self):
    check_every_epoch = self.config['check_every_epoch']

    fig = plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.plot(self.stats['iter_loss'],'oc')
    plt.plot(self.stats['avg_loss'],'-b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(132)
    plt.plot(self.stats['train_acc'],'-o', label='train')
    plt.plot(self.stats['val_acc'],'-o', label='validation')
    plt.xlabel(f'Every {check_every_epoch} Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(133)
    plt.plot(self.stats['ratio'],'bo', label='ratio')
    plt.yscale('log')
    plt.yticks([1e-2,1e-3,1e-4,1e-5,1e-6],['1e-2','1e-3','1e-4','1e-5','1e-6'])
    plt.xlabel(f'Every {check_every_epoch} Epoch')
    plt.legend()
    plt.show()

    return fig



class ClassifierSolver(Solver):
  def __init__(self, model, train_loader, val_loader, optimizer, lr_scheduler= None,  check_every_epoch=2, random_seed=0):
    super().__init__(model, train_loader, val_loader, optimizer, lr_scheduler, check_every_epoch, random_seed)

  def train_fn(self, Xtr, Ytr):

    with amp.autocast():
      y_pred = self.model(Xtr)
      loss = torch.nn.CrossEntropyLoss()(y_pred,Ytr)

    self.model.zero_grad(set_to_none=True)
    
    self.scaler.scale(loss).backward()
    self.scaler.step(self.optimizer)
    self.scaler.update()

    return loss.item(), y_pred

  @staticmethod
  def load_check_point(PATH, model, train_loader, val_loader, optimizer, lr, lr_scheduler= None):
    return Solver._load_check_point(ClassifierSolver, PATH, model, train_loader, val_loader, optimizer, lr, lr_scheduler)