import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Paper original helper function
def round_filters(filters, width_coefficient, depth_divisor= 8):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


class efficientNet (nn.Module):
  '''
  
  TODO:
    - Figure out how to scale stage 1 and stage 9. ANSWER: stem and top dont scale
    - Base model parameters number are now correct, need to check fi scaling rule. 
        -output channels is round or ceil?
    - There are dropout layer in implementation but not in papaer, ADDED

    - FIND drop rate details

  '''

  def __init__(self,fi=0, num_classes=1000):
    super().__init__()

    # alpha = 1.2 **fi    #Depth, num of layers
    # beta = 1.1 **fi     #Width, num of channels
    # gemma = 1.15 **fi   #Resolution, NOT USED in implementation, however, it roughly determine input resolution, B0: 224, B1: 240, B2: 260, B3: 300, B4: 380, B5: 456, B6: 528, B7: 600, L2: 800

    classifier_dropout = {0: 0.2, 1: 0.2, 2: 0.3, 3: 0.3, 4: 0.4, 5: 0.4, 6: 0.5, 7: 0.5, "L2": 0.5}
    width_multiplier = {0: 1, 1: 1, 2: 1.1, 3: 1.2, 4: 1.4, 5: 1.6, 6: 1.8, 7: 2, "L2": 4.3} # Beta ~= 1.1 ** fi
    depth_multiplier = {0: 1, 1: 1.1, 2: 1.2, 3: 1.4, 4: 1.8, 5: 2.2, 6: 2.6, 7: 3.1, "L2": 5.3} # Alpha ~= 1.2 ** fi

    stage_1_param = {"output_channels": round_filters(32, width_multiplier[fi]),
                     "kernel_size": 3}

    stage_2_param = {"output_channels": round_filters(16, width_multiplier[fi]),
                     "kernel_size": 3,
                     "channel_expand_factor": 1,
                     "num_layers": math.ceil(1 * depth_multiplier[fi]),
                     "stage_downsample": False}

    stage_3_param = {"output_channels": round_filters(24, width_multiplier[fi]),
                     "kernel_size": 3,
                     "channel_expand_factor": 6,
                     "num_layers": math.ceil(2 * depth_multiplier[fi]),
                     "stage_downsample": True}

    stage_4_param = {"output_channels": round_filters(40, width_multiplier[fi]),
                     "kernel_size": 5,
                     "channel_expand_factor": 6,
                     "num_layers": math.ceil(2 * depth_multiplier[fi]),
                     "stage_downsample": True}

    stage_5_param = {"output_channels": round_filters(80, width_multiplier[fi]),
                     "kernel_size": 3,
                     "channel_expand_factor": 6,
                     "num_layers": math.ceil(3 * depth_multiplier[fi]),
                     "stage_downsample": True}     

    stage_6_param = {"output_channels": round_filters(112, width_multiplier[fi]),
                     "kernel_size": 5,
                     "channel_expand_factor": 6,
                     "num_layers": math.ceil(3 * depth_multiplier[fi]),
                     "stage_downsample": False}

    stage_7_param = {"output_channels": round_filters(192, width_multiplier[fi]),
                     "kernel_size": 5,
                     "channel_expand_factor": 6,
                     "num_layers": math.ceil(4 * depth_multiplier[fi]),
                     "stage_downsample": True}

    stage_8_param = {"output_channels": round_filters(320, width_multiplier[fi]),
                     "kernel_size": 3,
                     "channel_expand_factor": 6,
                     "num_layers": math.ceil(1 * depth_multiplier[fi]),
                     "stage_downsample": False}

    stage_9_param = {"output_channels": round_filters(1280, width_multiplier[fi]),
                     "kernel_size": 1}

    #TODO
    drop_rate = 0.2

    stage_1 = nn.Sequential(nn.Conv2d(3,stage_1_param["output_channels"],stage_1_param["kernel_size"],stride=2, padding= 1, bias= False),
                            nn.BatchNorm2d(stage_1_param["output_channels"],momentum=0.01),
                            nn.ReLU())
    
    stage_2 = MBConv_stage(stage_1_param["output_channels"],stage_2_param["output_channels"],stage_2_param["kernel_size"],stage_2_param["channel_expand_factor"],stage_2_param["num_layers"],drop_rate, stage_2_param["stage_downsample"])

    stage_3 = MBConv_stage(stage_2_param["output_channels"],stage_3_param["output_channels"],stage_3_param["kernel_size"],stage_3_param["channel_expand_factor"],stage_3_param["num_layers"],drop_rate, stage_3_param["stage_downsample"])
                
    stage_4 = MBConv_stage(stage_3_param["output_channels"],stage_4_param["output_channels"],stage_4_param["kernel_size"],stage_4_param["channel_expand_factor"],stage_4_param["num_layers"],drop_rate, stage_4_param["stage_downsample"])
   
    stage_5 = MBConv_stage(stage_4_param["output_channels"],stage_5_param["output_channels"],stage_5_param["kernel_size"],stage_5_param["channel_expand_factor"],stage_5_param["num_layers"],drop_rate, stage_5_param["stage_downsample"])   
                              
    stage_6 = MBConv_stage(stage_5_param["output_channels"],stage_6_param["output_channels"],stage_6_param["kernel_size"],stage_6_param["channel_expand_factor"],stage_6_param["num_layers"],drop_rate, stage_6_param["stage_downsample"])
                              
    stage_7 = MBConv_stage(stage_6_param["output_channels"],stage_7_param["output_channels"],stage_7_param["kernel_size"],stage_7_param["channel_expand_factor"],stage_7_param["num_layers"],drop_rate, stage_7_param["stage_downsample"])
                              
    stage_8 = MBConv_stage(stage_7_param["output_channels"],stage_8_param["output_channels"],stage_8_param["kernel_size"],stage_8_param["channel_expand_factor"],stage_8_param["num_layers"],drop_rate, stage_8_param["stage_downsample"])
    
    stage_9 = nn. Sequential( nn.Conv2d(stage_8_param["output_channels"], stage_9_param["output_channels"], stage_9_param["kernel_size"], bias= False),
                              nn.BatchNorm2d(stage_9_param["output_channels"],momentum=0.01),
                              nn.ReLU(),
                              nn.AdaptiveAvgPool2d((1,1)),
                              nn.Flatten(),
                              nn.Dropout(classifier_dropout[fi]),
                              nn.Linear(stage_9_param["output_channels"], num_classes))
    
    self.model = nn.Sequential(stage_1,
                               stage_2,
                               stage_3,
                               stage_4,
                               stage_5,
                               stage_6,
                               stage_7,
                               stage_8,
                               stage_9)
  def forward(self,x):
      return self.model(x)

class SE_block (nn.Module):
  '''
  Squeeze and Excitation block
  Some channels may be more important than another, therefore create a weight for number
  of channels allow the network to prioritize different channels.
  Sigmoid is used to ensure the outputs are between 0 and 1.

  Inputs:
  - in_channel: num of channels for the input feature maps (after expand)
  - ratio: Used to reduce dimension in linear layers, original paper default value = 16
  - has_se: Equal Ture when MBConv kernel size is 5, otherwise False
  Outpus:
  - A weighted output with the same shape with input, if has_se == Flase, return identity

  Note:
  - Actual implementaion of SE paper use linear but EfficientNet used Conv2d with kernel size 1 with original input channels (before expand) * 0.25. 
    Here I convert in_channel back to number of channels before expansion and devide by 4 to match the implementation of EffcientNet as close as possible.
    However, I cant use Conv2d without major change after global average pooling. Therefore, I kept the linear layer.
    I suspect it make little difference between the two.
  '''
  def __init__(self,in_channel, channel_expand_factor, ratio= 4):
    super().__init__()

    intermediate_channel = int((in_channel / channel_expand_factor) // ratio)
    
    self.weights = nn.Sequential(
                                nn.AdaptiveAvgPool2d((1,1)),
                                nn.Flatten(),
                                nn.Linear(in_channel,intermediate_channel),
                                nn.ReLU(),
                                nn.Linear(intermediate_channel, in_channel),
                                nn.Sigmoid()
                                )

  def forward (self,x):
      return x * self.weights(x).unsqueeze(-1).unsqueeze(-1)


class MBConv_block(nn.Module):
  '''
  Inputs:
    - channel_expand_factor: channel expand coeffient within inverse bottleneck block
    - keepdim: if ture, keep dimention of input's width, and height the same for output
               if false, input height and width are downsample by a factor of 2 
  Note:
    - Convolution bias parameters are turned off for layer right before batchnorm, since batchnorm eliminate any biases affects
    - Actually implementation use Swish activation
  '''

  def __init__ (self, in_channel, out_channel, kernel_size, drop_rate_= 0.2, channel_expand_factor=6, keepdim=True):
    super().__init__()
    self.drop_rate_ = drop_rate_
    self.dimension_not_change = keepdim and in_channel == out_channel

    intermidiate_channels = in_channel*channel_expand_factor
    pad = (kernel_size - 1) // 2

    # if keepdim, image size are not changing
    stride = 1 if keepdim else 2

    self.inv_bottleneck_block = nn.Sequential(
                                              nn.Conv2d(in_channel, intermidiate_channels, kernel_size=1, bias=False),
                                              nn.BatchNorm2d(intermidiate_channels,momentum=0.01),
                                              nn.ReLU(),
                                              nn.Conv2d(intermidiate_channels, intermidiate_channels, kernel_size, stride, pad, groups=intermidiate_channels, bias= False), # Depthwise Conv
                                              nn.BatchNorm2d(intermidiate_channels,momentum=0.01),
                                              nn.ReLU(),
                                              SE_block(intermidiate_channels, channel_expand_factor),
                                              nn.Conv2d(intermidiate_channels, out_channel, kernel_size=1, bias=False),
                                              nn.BatchNorm2d(out_channel,momentum=0.01),# momentum match tensorflow default setting/ original paper implementation
                                              )

    self.survival_rate = 0.8
    self.bernoulli = torch.distributions.bernoulli.Bernoulli(self.survival_rate)

  def forward (self, x):

    # stochastic_depth_drop = (self.bernoulli.sample().item() == 0.)
    # Stochastic depth drop (only during training)
    # if self.training and self.dimension_not_change and stochastic_depth_drop:
    #   return x
    
    output = self.inv_bottleneck_block(x)

    # using dropout and residual connection when dimension is not changed
    if self.dimension_not_change:
      output = F.dropout(output, self.drop_rate_, self.training)  +  x
    
    # during inference will use all layers and down weighted the output by the survival rate
    # if not self.training:
    #   output *= self.survival_rate

    return output


class MBConv_stage (nn.Module):

  def __init__(self, in_channel_, out_channel_, kernel_size_, channel_expand_factor_, num_layers_, drop_rate_= 0.2, stage_downsample_= True):
    super().__init__()

    module_list = nn.ModuleList()

    for i in range(num_layers_):
      cin = in_channel_ if i == 0 else out_channel_                             # first case where input channels equal actual inputs, otherwise it equals to output channels
      keepdim = False if i == num_layers_ - 1 and stage_downsample_ else True   # last case where we want to downsample the image width and height by a factor of 2 if stage_downsample_ = True
      
      module_list.append(MBConv_block(cin, out_channel_, kernel_size_, drop_rate_, channel_expand_factor_, keepdim))

    self.stage = nn.Sequential(*module_list)
  def forward(self, x):
    return self.stage(x)
