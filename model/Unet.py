import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.utils import spectral_norm

def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
  else:
    block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
  if bn:
    block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
  if dropout:
    block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block

def blockUNet_D(in_c, out_c, name, transposed=False, bn=True, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s_conv' % name, spectral_norm(nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)))
  else:
    block.add_module('%s_tconv' % name, spectral_norm(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)))
  if bn:
    block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
  if dropout:
    block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block


class D(nn.Module):
  def __init__(self, nc, nf):
    super(D, self).__init__()

    self.layers = nn.ModuleList([
      spectral_norm(nn.Conv2d(nc, nf, 4, 2, 1, bias=False)),
      blockUNet_D(nf, nf*2, 'layer1', transposed=False, bn=False, relu=False, dropout=False),
      blockUNet_D(nf*2, nf*4, 'layer2', transposed=False, bn=False, relu=False, dropout=False),   #bn=True
      nn.LeakyReLU(0.2, inplace=True),
      spectral_norm(nn.Conv2d(nf*4, nf*8, 4, 1, 1, bias=False)),
      # nn.BatchNorm2d(nf*8),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(nf*8, 1, 4, 1, 1, bias=False), #spectral_norm
      nn.Sigmoid()
    ])

    """
    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s_conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    # 128
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 64
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 32    
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s_conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    main.add_module('%s_bn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1 
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s_conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    main.add_module('%s_sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main
    """
  def forward(self, x):
    # output = self.main(x)
    for layer in self.layers:
      x = layer(x)
    return x

class G(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    super(G, self).__init__()

    # input is 256 x 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    layer1 = nn.Sequential()
    layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
    # input is 128 x 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer2 = blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 64 x 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer3 = blockUNet(nf*2, nf*4, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer4 = blockUNet(nf*4, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 16
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer5 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 8
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer6 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 4
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer7 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 2 x  2
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer8 = blockUNet(nf*8, nf*8, name, transposed=False, bn=False, relu=False, dropout=False)

    ## NOTE: decoder
    # input is 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8
    dlayer8 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)

    #import pdb; pdb.set_trace()
    # input is 2
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer7 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
    # input is 4
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer6 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
    # input is 8
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer5 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 16
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer4 = blockUNet(d_inc, nf*4, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 32
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*4*2
    dlayer3 = blockUNet(d_inc, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 64
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*2*2
    dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 128
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    dlayer1 = nn.Sequential()
    d_inc = nf*2
    dlayer1.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    dlayer1.add_module('%s_tconv' % name, nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False))
    dlayer1.add_module('%s_tanh' % name, nn.Tanh())

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3
    self.layer4 = layer4
    self.layer5 = layer5
    self.layer6 = layer6
    self.layer7 = layer7
    self.layer8 = layer8
    self.dlayer8 = dlayer8
    self.dlayer7 = dlayer7
    self.dlayer6 = dlayer6
    self.dlayer5 = dlayer5
    self.dlayer4 = dlayer4
    self.dlayer3 = dlayer3
    self.dlayer2 = dlayer2
    self.dlayer1 = dlayer1

  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    out6 = self.layer6(out5)
    out7 = self.layer7(out6)
    out8 = self.layer8(out7)
    dout8 = self.dlayer8(out8)
    dout8_out7 = torch.cat([dout8, out7], 1)
    dout7 = self.dlayer7(dout8_out7)
    dout7_out6 = torch.cat([dout7, out6], 1)
    dout6 = self.dlayer6(dout7_out6)
    dout6_out5 = torch.cat([dout6, out5], 1)
    dout5 = self.dlayer5(dout6_out5)
    dout5_out4 = torch.cat([dout5, out4], 1)
    dout4 = self.dlayer4(dout5_out4)
    dout4_out3 = torch.cat([dout4, out3], 1)
    dout3 = self.dlayer3(dout4_out3)
    dout3_out2 = torch.cat([dout3, out2], 1)
    dout2 = self.dlayer2(dout3_out2)
    dout2_out1 = torch.cat([dout2, out1], 1)
    dout1 = self.dlayer1(dout2_out1)
    return dout1

# generator from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix


class UnetGenerator(nn.Module):
  """Create a Unet-based generator"""

  def __init__(self, input_nc, output_nc,ngf=64, num_downs=8, norm_layer=nn.BatchNorm2d, use_dropout=False):
    """Construct a Unet generator
    Parameters:
        input_nc (int)  -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                            image of size 128x128 will become of size 1x1 # at the bottleneck
        ngf (int)       -- the number of filters in the last conv layer
        norm_layer      -- normalization layer

    We construct the U-Net from the innermost layer to the outermost layer.
    It is a recursive process.
    """
    super(UnetGenerator, self).__init__()
    # construct unet structure
    unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
    for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
    # gradually reduce the number of filters from ngf * 8 to ngf
    unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
    self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

  def forward(self, input):
    """Standard forward"""
    return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
  """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
  """

  def __init__(self, outer_nc, inner_nc, input_nc=None,
                submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
    """Construct a Unet submodule with skip connections.

    Parameters:
        outer_nc (int) -- the number of filters in the outer conv layer
        inner_nc (int) -- the number of filters in the inner conv layer
        input_nc (int) -- the number of channels in input images/features
        submodule (UnetSkipConnectionBlock) -- previously defined submodules
        outermost (bool)    -- if this module is the outermost module
        innermost (bool)    -- if this module is the innermost module
        norm_layer          -- normalization layer
        use_dropout (bool)  -- if use dropout layers.
    """
    super(UnetSkipConnectionBlock, self).__init__()
    self.outermost = outermost
    use_bias = False
    if input_nc is None:
        input_nc = outer_nc
    downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                          stride=2, padding=1, bias=use_bias)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(inner_nc)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(outer_nc)

    if outermost:
        upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                    kernel_size=4, stride=2,
                                    padding=1)
        down = [downconv]
        up = [uprelu, upconv, nn.Tanh()]
        model = down + [submodule] + up
    elif innermost:
        upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                    kernel_size=4, stride=2,
                                    padding=1, bias=use_bias)
        down = [downrelu, downconv]
        up = [uprelu, upconv, upnorm]
        model = down + up
    else:
        upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                    kernel_size=4, stride=2,
                                    padding=1, bias=use_bias)
        down = [downrelu, downconv, downnorm]
        up = [uprelu, upconv, upnorm]

        if use_dropout:
            model = down + [submodule] + up + [nn.Dropout(0.5)]
        else:
            model = down + [submodule] + up

    self.model = nn.Sequential(*model)

  def forward(self, x):
    if self.outermost:
      return self.model(x)
    else:   # add skip connections
      return torch.cat([x, self.model(x)], 1)

