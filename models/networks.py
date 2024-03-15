import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

import functools
from einops import rearrange

import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from fightingcv_attention.attention.CBAM import CBAMBlock,CBAMECABlock
from fightingcv_attention.attention.SKAttention import SKAttention
from fightingcv_attention.attention.SEAttention import SEAttention
from fightingcv_attention.attention.ECAAttention import ECAAttention



###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    
    elif args.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1): # 有conv或者linear层
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'base_resnet18':
        net = ResNet(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'base_transformer_pos_s4':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned')

    elif args.net_G == 'base_transformer_pos_s4_dd8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8)

    # elif args.net_G == 'base_transformer_pos_s4fpn_diff_dd8_dedim8':
    #     net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
    #                          with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)
    elif args.net_G == 'base_transformer_pos_s4_diff_dd8_dedim8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8) 
    
    elif args.net_G == 'base_transformer_pos_s4fpn_diff_dd8_e4d4':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=4, dec_depth=4, decoder_dim_head=8) 

    elif args.net_G == 'base_transformer_pos_s4fpn_diff_dd8_e2d6':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)       

    elif args.net_G == 'base_transformer_pos_s5fpn_diff_dd8_e2d6':
        net = BASE_Transformer_S5(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=5,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)         

    elif args.net_G == 'base_transformer_pos_s3fpn_diff_dd8_e2d6':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=3,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)
    
    elif args.net_G == 'base_transformer_pos_s2fpn_diff_dd8_e2d6':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=2,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)
                                       
    elif args.net_G == 'base_transformer_pos_s4fpn_dd8_e2d6':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)

    elif args.net_G == 'base_transformer_pos_s4_dd8_e2d6':
        net = BASE_Transformer_NOFPN(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)                         
    elif args.net_G == 'base_transformer_pos_s4_diff_dd8_e2d6':
        net = BASE_Transformer_NOFPN(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)  
    
    elif args.net_G == 'base_transformer_pos_s4fpn_diff_dd8_e6d2':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=6, dec_depth=2, decoder_dim_head=8) 
                             

    elif args.net_G == 'base_transformer_pos_s4fpn_diff_dd8_e6d2_sk':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=6, dec_depth=2, decoder_dim_head=8)

    elif args.net_G == 'base_transformer_pos_s4fpn_diff_dd8_e8d1':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=8, dec_depth=1, decoder_dim_head=8)

    elif args.net_G == 'base_transformer_pos_s4fpn_diff_dd8_e1d8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)

    elif args.net_G == 'base_transformer_pos_s4fpn_diff_dd8_e2d6_sk':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)

    elif args.net_G == 'base_transformer_pos_s5fpn_diff_dd8_e2d6_sk':
        net = BASE_Transformer_S5(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=5,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)

    elif args.net_G == 'base_transformer_pos_s5fpn_diff_dd8_e2d6_se':
        net = BASE_Transformer_S5(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=5,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)

    elif args.net_G == 'base_transformer_pos_s5fpn_diff_dd8_e2d6_eca':
        net = BASE_Transformer_S5(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=5,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)

    elif args.net_G == 'base_transformer_pos_s5fpn_diff_dd8_e2d6_cbam':
        net = BASE_Transformer_S5(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=5,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)

    elif args.net_G == 'base_transformer_pos_s5fpn_diff_dd8_e2d6_cbameca':
        net = BASE_Transformer_S5(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=5,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)

    elif args.net_G == 'base_transformer_pos_s4fpn_diff_dd8_e2d6_se':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)

    elif args.net_G == 'base_transformer_pos_s4fpn_diff_dd8_e2d6_cbam':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)             


    elif args.net_G == 'base_transformer_pos_s4fpn_diff_dd8_e2d6_cbameca':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)  

    elif args.net_G == 'base_transformer_pos_s4fpn_diff_dd8_e2d6_eca':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=2, dec_depth=6, decoder_dim_head=8)                  

    # elif args.net_G == 'base_transformer_pos_s5fpn_diff_dd8_e6d2':
    #     net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=5,
    #                          with_pos='learned', enc_depth=6, dec_depth=2, decoder_dim_head=8) 

    # elif args.net_G == 'base_transformer_pos_s3fpn_diff_dd8_e6d2':
    #     net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=3,
    #                          with_pos='learned', enc_depth=6, dec_depth=2, decoder_dim_head=8) 
    # elif args.net_G == 'base_transformer_pos_s2fpn_diff_dd8_e6d2':
    #     net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=2,
    #                          with_pos='learned', enc_depth=6, dec_depth=2, decoder_dim_head=8) 
    # ########### 添加了cbameca,改成了bi                                          
    # elif args.net_G == 'base_transformer_pos_s4fpn_diff_dd8_e6d2bi_cbameca':
    #     net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
    #                          with_pos='learned', enc_depth=6, dec_depth=2, decoder_dim_head=8)
    ########### 添加了cbameca, not bi                              
    elif args.net_G == 'base_transformer_pos_s4fpn_diff_dd8_e6d2_cbameca':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=6, dec_depth=2, decoder_dim_head=8)
    # ########### 添加了cbameca(before diff), not bi                             
    # elif args.net_G == 'base_transformer_pos_s4fpn_diff_dd8_e6d2_cbameca_bd':
    #     net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
    #                          with_pos='learned', enc_depth=6, dec_depth=2, decoder_dim_head=8)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num


        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        ############ my     
        elif self.resnet_stages_num == 2:
            layers = 64
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        xout = [x]
        # print('after cov1:', x.size())
        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        xout.append(x_4)
        # print('after cov2:', x_4.size())
        ###########s2 去掉conv3
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        xout.append(x_8)
        # print('after cov3:', x_8.size())

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256
            xout.append(x_8)
            # print('after cov4:', x_8.size())

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
            xout.append(x_8)
            # print('after cov5:', x_8.size())
            
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        # if self.if_upsample_2x:
        #     x = self.upsamplex2(x_8)
        #     # print('if_upsample_2x:', x.size())
        # else:
        #     x = x_8
        #     # print('ifnot_upsample_2x:', x.size())
        
        if self.if_upsample_2x:
            x_nofpn = self.upsamplex2(x_8)
            # print('if_upsample_2x:', x.size())
        else:
            x_nofpn = x_8
            # print('ifnot_upsample_2x:', x.size())

        x = x_nofpn ######### without   fpn
        x = self.conv_pred(x)
        return x

class ResNet_FPN_S4(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet_FPN_S4, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.fpn = FPN_S4()  #####satge = 4
        # self.fpn2 = FPN_S5(out_channels=512)  #####satge = 5
        # self.fpn3 = FPN_S3(out_channels=128)  #####satge = 3
        # self.fpn4 = FPN_S2(out_channels=64)  #####satge = 2

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        ############ my     
        elif self.resnet_stages_num == 2:
            layers = 64
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        xout = [x]
        # print('after cov1:', x.size())

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        xout.append(x_4)
        # print('after cov2:', x_4.size())
        # ###########
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        xout.append(x_8)
        # print('after cov3:', x_8.size())


        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256
            xout.append(x_8)
            # print('after cov4:', x_8.size())

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
            xout.append(x_8)
            # print('after cov5:', x_8.size())
            
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
            # print('if_upsample_2x:', x.size())
        else:
            x = x_8
            # print('ifnot_upsample_2x:', x.size())
        
        # if self.if_upsample_2x:
        #     x_nofpn = self.upsamplex2(x_8)
        #     # print('if_upsample_2x:', x.size())
        # else:
        #     x_nofpn = x_8
        #     # print('ifnot_upsample_2x:', x.size())

        # x = x_nofpn ######### without   fpn
        x = self.fpn(xout)  ############四层特征融合 [16,256,64,64]
        # x = self.fpn2(xout)  ############5层特征融合  [16,512,64,64]
        # x = self.fpn3(xout)  ############3层特征融合 [16,128,64,64]
        # x = self.fpn4(xout)  ############2层特征融合 [16,64,64,64]
        # print('after fpn:', x.size())
        # print(1/0)
        # output layers
        x = self.conv_pred(x)
        return x

class ResNet_FPN_S5(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet_FPN_S5, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        # self.fpn = FPN_S4()  #####satge = 4
        self.fpn2 = FPN_S5(out_channels=512)  #####satge = 5
        # self.fpn3 = FPN_S3(out_channels=128)  #####satge = 3
        # self.fpn4 = FPN_S2(out_channels=64)  #####satge = 2

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        ############ my     
        elif self.resnet_stages_num == 2:
            layers = 64
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        xout = [x]
        # print('after cov1:', x.size())

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        xout.append(x_4)
        # print('after cov2:', x_4.size())
        # ###########
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        xout.append(x_8)
        # print('after cov3:', x_8.size())


        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256
            xout.append(x_8)
            # print('after cov4:', x_8.size())

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
            xout.append(x_8)
            # print('after cov5:', x_8.size())
            
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
            # print('if_upsample_2x:', x.size())
        else:
            x = x_8
            # print('ifnot_upsample_2x:', x.size())
        
        # if self.if_upsample_2x:
        #     x_nofpn = self.upsamplex2(x_8)
        #     # print('if_upsample_2x:', x.size())
        # else:
        #     x_nofpn = x_8
        #     # print('ifnot_upsample_2x:', x.size())

        # x = x_nofpn ######### without   fpn
        # x = self.fpn(xout)  ############四层特征融合 [16,256,64,64]
        x = self.fpn2(xout)  ############5层特征融合  [16,512,64,64]
        # x = self.fpn3(xout)  ############3层特征融合 [16,128,64,64]
        # x = self.fpn4(xout)  ############2层特征融合 [16,64,64,64]
        # print('after fpn:', x.size())
        # print(1/0)
        # output layers
        x = self.conv_pred(x)
        return x

class BASE_Transformer(ResNet_FPN_S4):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_Transformer, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer  ####
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos is 'learned':######可训练的位置编码
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
            # self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        self.diff = Diff(in_channels= 2* dim, out_channels=dim)
        
        ###########SK
        # self.sk = SKAttention(channel=32,reduction=4)
        # self.se = SEAttention(channel=32,reduction=4)
        # self.cbam = CBAMBlock(channel=32,reduction=4,kernel_size=7)
        # self.cbam_eca = CBAMECABlock(kernel_size_ca=3, kernel_size_sa=7)
        # self.eca = ECAAttention(kernel_size=3)
        
    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()##### b, l, n 
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()  ####b, c, n 
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x) 

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode is 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode is 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):#####encoder
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x
        
    def _forward_transformer_bi(self, x):##### bi encoder
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m): ########### transformer_decoder
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):  ###########简单decoder（不用transformer_decoder）
        b, c, h, w = x.shape
        b, l, c = m.shape  
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
            
        # forward transformer encoder  拼接
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)

        # # forward transformer encoder 分别进行
        # if self.token_trans:
            
        #     token1 = self._forward_transformer_bi(token1)
        #     token2 = self._forward_transformer_bi(token2)


        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        

        # print('x1.size:',x1.size())
        # print('x2.size:',x2.size())
        # print(1/0)

        # cbameca
        # x1 = self.cbam_eca(x1)
        # x2 = self.cbam_eca(x2)

        # #############sk
        # x1 = self.sk(x1)
        # x2 = self.sk(x2)

        # # #############se
        # x1 = self.se(x1)
        # x2 = self.se(x2)

        # # #############CBAM
        # x1 = self.cbam(x1)
        # x2 = self.cbam(x2)

        # #############eca
        # x1 = self.eca(x1)
        # x2 = self.eca(x2)
        
        # feature differencing
        # x = torch.abs(x1 - x2)
        x = self.diff(torch.cat((x1, x2), dim=1))

        #############cbam模块
        # x = self.cbam_eca(x)

        if not self.if_upsample_2x:######## why上采样？？？？？？
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        # forward small cnn
        x = self.classifier(x)  ####分类器，两个conv层
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

class BASE_Transformer_S5(ResNet_FPN_S5):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_Transformer_S5, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer  ####
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos is 'learned':######可训练的位置编码
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
            # self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        self.diff = Diff(in_channels= 2* dim, out_channels=dim)
        
        ###########SK
        # self.sk = SKAttention(channel=32,reduction=4)
        # self.se = SEAttention(channel=32,reduction=4)
        # self.cbam = CBAMBlock(channel=32,reduction=4,kernel_size=7)
        self.cbam_eca = CBAMECABlock(kernel_size_ca=3, kernel_size_sa=7)
        # self.eca = ECAAttention(kernel_size=3)
        
    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()##### b, l, n 
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()  ####b, c, n 
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x) 

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode is 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode is 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):#####encoder
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x
        
    def _forward_transformer_bi(self, x):##### bi encoder
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m): ########### transformer_decoder
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):  ###########简单decoder（不用transformer_decoder）
        b, c, h, w = x.shape
        b, l, c = m.shape  
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
            
        # forward transformer encoder  拼接
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)

        # # forward transformer encoder 分别进行
        # if self.token_trans:
            
        #     token1 = self._forward_transformer_bi(token1)
        #     token2 = self._forward_transformer_bi(token2)


        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        

        # print('x1.size:',x1.size())
        # print('x2.size:',x2.size())
        # print(1/0)

        # #############sk
        # x1 = self.sk(x1)
        # x2 = self.sk(x2)

        # # #############se
        # x1 = self.se(x1)
        # x2 = self.se(x2)

        # # #############CBAM
        # x1 = self.cbam(x1)
        # x2 = self.cbam(x2)

        ################cbameca
        x1 = self.cbam_eca(x1)
        x2 = self.cbam_eca(x2)

        # #############eca
        # x1 = self.eca(x1)
        # x2 = self.eca(x2)
        
        # feature differencing
        # x = torch.abs(x1 - x2)
        x = self.diff(torch.cat((x1, x2), dim=1))

        #############cbam模块
        # x = self.cbam_eca(x)

        if not self.if_upsample_2x:######## why上采样？？？？？？
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        # forward small cnn
        x = self.classifier(x)  ####分类器，两个conv层
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

class BASE_Transformer_NOFPN(ResNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_Transformer_NOFPN, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer  ####
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos is 'learned':######可训练的位置编码
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
            # self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

        self.diff = Diff(in_channels= 2* dim, out_channels=dim)
        
        ###########SK
        # self.sk = SKAttention(channel=32,reduction=4)
        # self.se = SEAttention(channel=32,reduction=4)
        # self.cbam = CBAMBlock(channel=32,reduction=4,kernel_size=7)
        # self.cbam_eca = CBAMECABlock(kernel_size_ca=3, kernel_size_sa=7)
        # self.eca = ECAAttention(kernel_size=3)
        
    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()##### b, l, n 
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()  ####b, c, n 
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x) 

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode is 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode is 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):#####encoder
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x
        
    def _forward_transformer_bi(self, x):##### bi encoder
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m): ########### transformer_decoder
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):  ###########简单decoder（不用transformer_decoder）
        b, c, h, w = x.shape
        b, l, c = m.shape  
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
            
        # forward transformer encoder  拼接
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)

        # # forward transformer encoder 分别进行
        # if self.token_trans:
            
        #     token1 = self._forward_transformer_bi(token1)
        #     token2 = self._forward_transformer_bi(token2)


        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        

        # print('x1.size:',x1.size())
        # print('x2.size:',x2.size())
        # print(1/0)

        # cbameca
        # x1 = self.cbam_eca(x1)
        # x2 = self.cbam_eca(x2)

        # #############sk
        # x1 = self.sk(x1)
        # x2 = self.sk(x2)

        # # #############se
        # x1 = self.se(x1)
        # x2 = self.se(x2)

        # # #############CBAM
        # x1 = self.cbam(x1)
        # x2 = self.cbam(x2)

        # #############eca
        # x1 = self.eca(x1)
        # x2 = self.eca(x2)
        
        # feature differencing
        # x = torch.abs(x1 - x2)
        x = self.diff(torch.cat((x1, x2), dim=1))

        #############cbam模块
        # x = self.cbam_eca(x)

        if not self.if_upsample_2x:######## why上采样？？？？？？
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        # forward small cnn
        x = self.classifier(x)  ####分类器，两个conv层
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

#Difference module
# def conv_diff(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.BatchNorm2d(out_channels),
#         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#         nn.ReLU()
#     )
class Diff(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.con1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.con2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.con1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.con2(x)
        x = self.act(x)

        return x


class FPN_S4(nn.Module):
    """resnet(S4) + FPN

    return: 融合后的特征层
    """
    def __init__(self, in_channels = [64, 64, 128, 256], out_channels = 256):
        super().__init__()

        self.conv1by1_4 = nn.Conv2d(in_channels[-1], out_channels, 1)
        self.conv1by1_3 = nn.Conv2d(in_channels[-2], out_channels, 1)
        self.conv1by1_2 = nn.Conv2d(in_channels[-3], out_channels, 1)
        self.conv1by1_1 = nn.Conv2d(in_channels[-4], out_channels, 1)


    def forward(self, x):
        f4 = x[-1]  #####[16,256,32,32]  (no upsampling)
        f3 = x[-2]  #####[16,128,32,32]
        f2 = x[-3]  #####[16,64,64,64]
        f1 = x[-4]  #####[16,64,64,64]
    

        # f4 = self.conv1by1_4(f4) 
        f4 = F.interpolate(f4, scale_factor= 2 , mode="nearest")  #####[16,256,64,64]  
        
        f3 = F.interpolate(f3, scale_factor= 2 , mode="nearest")  #####[16,128,64,64]
        f3 = self.conv1by1_3(f3) + f4                             #####[16,256,64,64]  
        # f3 = F.interpolate(f3, scale_factor= 2 , mode="nearest")

        f2 = self.conv1by1_2(f2) + f3
        # f2 = F.interpolate(f2, scale_factor= 2 , mode="nearest")

        f1 = self.conv1by1_1(f1) + f2

        return f1

class FPN_S5(nn.Module):
    """resnet(S5) + FPN

    return: 融合后的特征层
    """
    def __init__(self, in_channels = [64, 64, 128, 256, 512], out_channels = 256):
        super().__init__()

        self.conv1by1_5 = nn.Conv2d(in_channels[-1], out_channels, 1)
        self.conv1by1_4 = nn.Conv2d(in_channels[-2], out_channels, 1)
        self.conv1by1_3 = nn.Conv2d(in_channels[-3], out_channels, 1)
        self.conv1by1_2 = nn.Conv2d(in_channels[-4], out_channels, 1)
        self.conv1by1_1 = nn.Conv2d(in_channels[-5], out_channels, 1)


    def forward(self, x):
        f5 = x[-1]  #####[16,512,32,32]  (no upsampling)
        f4 = x[-2]  #####[16,256,32,32]  (no upsampling)
        f3 = x[-3]  #####[16,128,32,32]
        f2 = x[-4]  #####[16,64,64,64]
        f1 = x[-5]  #####[16,64,64,64]
    

        # f4 = self.conv1by1_4(f4) 
        
        f4 = f5 + self.conv1by1_4(f4) #####[16,512,32,32] 
        f4 = F.interpolate(f4, scale_factor= 2 , mode="nearest")  #####[16,512,64,64]  
        
        f3 = F.interpolate(f3, scale_factor= 2 , mode="nearest")  #####[16,128,64,64]
        f3 = self.conv1by1_3(f3) + f4                             #####[16,512,64,64]  
        # f3 = F.interpolate(f3, scale_factor= 2 , mode="nearest")

        f2 = self.conv1by1_2(f2) + f3
        # f2 = F.interpolate(f2, scale_factor= 2 , mode="nearest")

        f1 = self.conv1by1_1(f1) + f2
        # print("f1hou:", f1.size())

        return f1


class FPN_S3(nn.Module):
    """resnet(S3) + FPN

    return: 融合后的特征层
    """
    def __init__(self, in_channels = [64, 64, 128], out_channels = 256):
        super().__init__()

        self.conv1by1_3 = nn.Conv2d(in_channels[-1], out_channels, 1)
        self.conv1by1_2 = nn.Conv2d(in_channels[-2], out_channels, 1)
        self.conv1by1_1 = nn.Conv2d(in_channels[-3], out_channels, 1)


    def forward(self, x):
        f3 = x[-1]  #####[16,128,32,32]
        f2 = x[-2]  #####[16,64,64,64]
        f1 = x[-3]  #####[16,64,64,64]
    
        
        f3 = F.interpolate(f3, scale_factor= 2 , mode="nearest")  #####[16,128,64,64]
        f3 = self.conv1by1_3(f3)                            #####[16,256,64,64]  

        f2 = self.conv1by1_2(f2) + f3
        # f2 = F.interpolate(f2, scale_factor= 2 , mode="nearest")

        f1 = self.conv1by1_1(f1) + f2
        # print("f1hou:", f1.size())

        return f1        

class FPN_S2(nn.Module):
    """resnet(S2) + FPN

    return: 融合后的特征层
    """
    def __init__(self, in_channels = [64, 64], out_channels = 256):
        super().__init__()

        self.conv1by1_2 = nn.Conv2d(in_channels[-1], out_channels, 1)
        self.conv1by1_1 = nn.Conv2d(in_channels[-2], out_channels, 1)


    def forward(self, x):
        f2 = x[-1]  #####[16,64,64,64]
        f1 = x[-2]  #####[16,64,64,64]

        f1 = self.conv1by1_1(f1) + self.conv1by1_2(f2)
    
        return f1        