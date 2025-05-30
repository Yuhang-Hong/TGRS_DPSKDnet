import torch
import torch.nn as nn
import torch.nn.functional as F
from .morph_layers2D_torch import * 


class MorphNet(nn.Module):
    def __init__(self, inchannel):
        super(MorphNet, self).__init__()
        num = 1
        kernel_size = 3
        self.conv1 = nn.Conv2d(
            inchannel, num, kernel_size=1, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.Erosion2d_1 = Erosion2d(num, num, kernel_size, soft_max=False)
        self.Dilation2d_1 = Dilation2d(num, num, kernel_size, soft_max=False)
        self.Erosion2d_2 = Erosion2d(num, num, kernel_size, soft_max=False)
        self.Dilation2d_2 = Dilation2d(num, num, kernel_size, soft_max=False)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        xop_2 = self.Dilation2d_1(self.Erosion2d_1(x))
        xcl_2 = self.Erosion2d_2(self.Dilation2d_2(x))
        x_top = x - xop_2
        x_blk = xcl_2 - x
        x_morph = torch.cat((x_top, x_blk, xop_2, xcl_2), 1)

        return x_morph


class Dynamic_perturbation_destylization(nn.Module):
    
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        # 自适应混合参数
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习混合系数
        self.gamma = nn.Parameter(torch.tensor(0.1))  # 扰动强度系数

    def forward(self, x):
        if self.training:
            N, C, H, W = x.shape
            
            # 1. 动态混合均值和方差
            idx_swap = torch.randperm(N)
            x_mix = self.alpha * x + (1 - self.alpha) * x[idx_swap]
            
            # 2. 对抗式扰动（沿梯度方向）
            delta = torch.randn_like(x) * self.gamma
            delta.requires_grad_()
            x_perturbed = x_mix + delta
            
            # 3. 更新运行统计量（模拟BN）
            mean = x_perturbed.mean([0, 2, 3])
            var = x_perturbed.var([0, 2, 3])
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            # 4. 标准化
            x = (x_perturbed - self.running_mean[None, :, None, None]) / \
                (self.running_var[None, :, None, None] + self.eps).sqrt()
            
        return x
    




class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),)+self.shape)


class Generator(nn.Module):
    def __init__(self, n=16, kernelsize=3, imdim=3, imsize=[13, 13], zdim=10, device=0):
        ''' w_ln 局部噪声权重
        '''
        super().__init__()
        stride = (kernelsize-1)//2
        self.zdim = zdim  # 潜在向量的维度，通常是生成器输入的随机噪声向量的大小。
        self.imdim = imdim
        self.imsize = imsize
        self.device = device
        num_morph = 4
        self.Morphology = MorphNet(imdim)

        self.conv_spa1 = nn.Conv2d(imdim, 3, 1, 1)
        self.conv_spa2 = nn.Conv2d(3, n, 1, 1)
        self.conv_spe1 = nn.Conv2d(imdim, n, imsize[0], 1)
        self.conv_spe2 = nn.ConvTranspose2d(n, n, imsize[0])
        self.conv1 = nn.Conv2d(n+n+num_morph, n, kernelsize, 1, stride)


        self.dpd1 = Dynamic_perturbation_destylization(num_features=n)
        self.dpd2 = Dynamic_perturbation_destylization(num_features=3)
        self.conv2 = nn.Conv2d(n, imdim, 1, 1)
    def forward(self, x):

        x_morph = self.Morphology(x) # 256,4,13,13

        x_spa = F.relu(self.conv_spa1(x))

        x_spe = F.relu(self.conv_spe1(x)) # 256,64,1,1

        x_spa = self.dpd2(x_spa)
        x_spa = self.conv_spa2(x_spa)
        x_spe = self.dpd1(x_spe)
        x_spe = self.conv_spe2(x_spe)
        x = F.relu(self.conv1(torch.cat((x_spa, x_spe, x_morph), 1)))
        x = torch.sigmoid(self.conv2(x))
        return x


if __name__ == "__main__":
    
    adaptive_randomizer = Dynamic_perturbation_destylization(num_features=64)

    # 前向传播
    x = torch.randn(32, 64, 13, 13)  # 模拟输入
    x_adapt = adaptive_randomizer(x)
    print(x_adapt.shape)