from typing import Union, Tuple, Optional
from involution import Involution2d
import torch
import torch.nn as nn
from torch.nn import functional as F
OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: Involution2d(C, C, kernel_size=3, stride=stride, padding=1),
  'sep_conv_5x5' : lambda C, stride, affine: Involution2d(C, C, kernel_size=5, stride=stride, padding=2),
  'sep_conv_7x7' : lambda C, stride, affine: Involution2d(C, C, kernel_size=7, stride=stride, padding=3),
  'dil_conv_3x3' : lambda C, stride, affine: Involution2d(C, C, kernel_size=3, stride=stride, padding=2, dilation=2),
  'dil_conv_5x5' : lambda C, stride, affine: Involution2d(C, C, kernel_size=5, stride=stride, padding=4, dilation=2),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

# class Involution2d(nn.Module):

#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  sigma_mapping: Optional[nn.Module] = None,
#                  kernel_size: Union[int, Tuple[int, int]] = (7, 7),
#                  stride: Union[int, Tuple[int, int]] = (1, 1),
#                  groups: int = 1,
#                  reduce_ratio: int = 1,
#                  dilation: Union[int, Tuple[int, int]] = (1, 1),
#                  padding: Union[int, Tuple[int, int]] = (3, 3),
#                  bias: bool = False,
#                  force_shape_match: bool = False,
#                  **kwargs) -> None:
#         """
#         Constructor method
#         :param in_channels: (int) Number of input channels
#         :param out_channels: (int) Number of output channels
#         :param sigma_mapping: (nn.Module) Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized
#         :param kernel_size: (Union[int, Tuple[int, int]]) Kernel size to be used
#         :param stride: (Union[int, Tuple[int, int]]) Stride factor to be utilized
#         :param groups: (int) Number of groups to be employed
#         :param reduce_ratio: (int) Reduce ration of involution channels
#         :param dilation: (Union[int, Tuple[int, int]]) Dilation in unfold to be employed
#         :param padding: (Union[int, Tuple[int, int]]) Padding to be used in unfold operation
#         :param bias: (bool) If true bias is utilized in each convolution layer
#         :param force_shape_match: (bool) If true potential shape mismatch is solved by performing avg pool
#         :param **kwargs: Unused additional key word arguments
#         """
#         # Call super constructor
#         super(Involution2d, self).__init__()
#         # Check parameters
#         assert isinstance(in_channels, int) and in_channels > 0, "in channels must be a positive integer."
#         assert in_channels % groups == 0, "out_channels must be divisible by groups"
#         assert isinstance(out_channels, int) and out_channels > 0, "out channels must be a positive integer."
#         assert out_channels % groups == 0, "out_channels must be divisible by groups"
#         assert isinstance(sigma_mapping, nn.Module) or sigma_mapping is None, \
#             "Sigma mapping must be an nn.Module or None to utilize the default mapping (BN + ReLU)."
#         assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), \
#             "kernel size must be an int or a tuple of ints."
#         assert isinstance(stride, int) or isinstance(stride, tuple), \
#             "stride must be an int or a tuple of ints."
#         assert isinstance(groups, int), "groups must be a positive integer."
#         assert isinstance(reduce_ratio, int) and reduce_ratio > 0, "reduce ratio must be a positive integer."
#         assert isinstance(dilation, int) or isinstance(dilation, tuple), \
#             "dilation must be an int or a tuple of ints."
#         assert isinstance(padding, int) or isinstance(padding, tuple), \
#             "padding must be an int or a tuple of ints."
#         assert isinstance(bias, bool), "bias must be a bool"
#         assert isinstance(force_shape_match, bool), "force shape match flag must be a bool"
#         # Save parameters
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
#         self.stride = stride if isinstance(stride, tuple) else (stride, stride)
#         self.groups = groups
#         self.reduce_ratio = reduce_ratio
#         self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
#         self.padding = padding if isinstance(padding, tuple) else (padding, padding)
#         self.bias = bias
#         self.force_shape_match = force_shape_match
#         # Init modules
#         self.sigma_mapping = sigma_mapping if sigma_mapping is not None else nn.Sequential(
#             nn.BatchNorm2d(num_features=self.out_channels // self.reduce_ratio, momentum=0.3), nn.ReLU())
#         self.initial_mapping = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
#                                          kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
#                                          bias=bias) if self.in_channels != self.out_channels else nn.Identity()
#         self.o_mapping = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
#         self.reduce_mapping = nn.Conv2d(in_channels=self.in_channels,
#                                         out_channels=self.out_channels // self.reduce_ratio, kernel_size=(1, 1),
#                                         stride=(1, 1), padding=(0, 0), bias=bias)
#         self.span_mapping = nn.Conv2d(in_channels=self.out_channels // self.reduce_ratio,
#                                       out_channels=self.kernel_size[0] * self.kernel_size[1] * self.groups,
#                                       kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias)
#         self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=dilation, padding=padding, stride=stride)

#     def __repr__(self) -> str:
#         """
#         Method returns information about the module
#         :return: (str) Info string
#         """
#         return ("{}({}, {}, kernel_size=({}, {}), stride=({}, {}), padding=({}, {}), "
#                 "groups={}, reduce_ratio={}, dilation=({}, {}), bias={}, sigma_mapping={})".format(
#             self.__class__.__name__,
#             self.in_channels,
#             self.out_channels,
#             self.kernel_size[0],
#             self.kernel_size[1],
#             self.stride[0],
#             self.stride[1],
#             self.padding[0],
#             self.padding[1],
#             self.groups,
#             self.reduce_mapping,
#             self.dilation[0],
#             self.dilation[1],
#             self.bias,
#             str(self.sigma_mapping)
#         ))

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass
#         :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, height, width]
#         :return: (torch.Tensor) Output tensor of the shape [batch size, out channels, height, width] (w/ same padding)
#         """
#         # Check input dimension of input tensor
#         assert input.ndimension() == 4, \
#             "Input tensor to involution must be 4d but {}d tensor is given".format(input.ndimension())
#         # Save input shape and compute output shapes
#         batch_size, _, in_height, in_width = input.shape
#         out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) \
#                      // self.stride[0] + 1
#         out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) \
#                     // self.stride[1] + 1
#         # Unfold and reshape input tensor
#         input_unfolded = self.unfold(self.initial_mapping(input))
#         input_unfolded = input_unfolded.view(batch_size, self.groups, self.out_channels // self.groups,
#                                              self.kernel_size[0] * self.kernel_size[1],
#                                              out_height, out_width)
#         # Reshape input to avoid shape mismatch problems
#         if self.force_shape_match:
#             input = F.adaptive_avg_pool2d(input,(out_height,out_width))
#         # Generate kernel
#         kernel = self.span_mapping(self.sigma_mapping(self.reduce_mapping(self.o_mapping(input))))
#         kernel = kernel.view(batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1],
#                              kernel.shape[-2], kernel.shape[-1]).unsqueeze(dim=2)
#         # Apply kernel to produce output
#         output = (kernel * input_unfolded).sum(dim=3)
#         # Reshape output
#         output = output.view(batch_size, -1, output.shape[-2], output.shape[-1])
#         return output