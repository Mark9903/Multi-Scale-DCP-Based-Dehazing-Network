"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: model.py
about: model for GridDehazeNet
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import cv2
import torchvision
import numpy
from residual_dense_block import RDB
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

# --- Downsampling block in GridDehazeNet  --- #
class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels, stride*in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out


# --- Upsampling block in GridDehazeNet  --- #
class UpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=stride, padding=1)
        self.conv = nn.Conv2d(in_channels, in_channels // stride, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x, output_size):
        out = F.relu(self.deconv(x, output_size=output_size))
        out = F.relu(self.conv(out))
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def GetDarkChannel(img, sz):
    device = img.device
    for i in range(0, (img.shape[1] - 1) // 16 + 1):
        tmpx, tmpy = torch.min(img[:,i * 16:i * 16 + 16,:,:,], dim = 1)
        tmpx = tmpx.unsqueeze(1)
        if i == 0:
            dc = tmpx
        else:
            dc = torch.cat((dc, tmpx), dim = 1)
            
    unloader = torchvision.transforms.ToPILImage()
    
    for i in range(0, dc.shape[0]):
        
        for j in range(0, dc.shape[1]):
            im = dc[i,j,:,:]
            im = unloader(im)
            im = numpy.array(im)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
            im = cv2.erode(im, kernel)
            im = Image.fromarray(im.astype('uint8'))
            transform_im = Compose([ToTensor(), Normalize((0.5), (0.5))])
            im = transform_im(im)
            im = im.unsqueeze(0)
            if j == 0:
                ans1 = im
            else:
                ans1 = torch.cat((ans1, im), dim = 1)
                
        if i == 0:
            ans2 = ans1
        else:
            ans2 = torch.cat((ans2, ans1), dim = 0)
    
    ans2 = ans2.to(device)
    
    return ans2

# --- Main model  --- #
class GridDehazeNet(nn.Module):
    def __init__(self, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(GridDehazeNet, self).__init__()
        self.rdb_module = nn.ModuleDict()
        self.orderModule = nn.ModuleDict()
        self.upsample_module = nn.ModuleDict()
        self.downsample_module = nn.ModuleDict()
        self.darkdownsample1 = DownSample(1)
        self.darkdownsample2 = DownSample(2)
        self.dark_rdb_module = nn.ModuleDict()
        self.dark_upsample_module = nn.ModuleDict()
        self.dark_downsample_module = nn.ModuleDict()
        self.height = height
        self.width = width
        self.stride = stride
        self.depth_rate = depth_rate
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((height, width, 2, depth_rate*stride**(height-1)))), requires_grad=attention)
        self.dark_coefficient = nn.Parameter(torch.Tensor(np.ones((height, width, 2, 20))), requires_grad=attention)
        self.dark_conv_in = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, 8, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.finalConv = nn.Conv2d(9, 3, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb_out = RDB(depth_rate, num_dense_layer, growth_rate)
        self.dark_rdb_in = RDB(1, 4, 1)
        self.dark_rdb_out = RDB(1, 4, 1)
        # extra
        # self.grad1 = nn.Sequential(
        #     BasicConv2d(in_channels, in_channels, kernel_size = (1, 3), padding = (0, 1)),
        #     BasicConv2d(in_channels, in_channels, kernel_size = (3, 1), padding = (1, 0)),
        #     BasicConv2d(in_channels, in_channels, kernel_size = 3, padding = 3, dilation = 3)
        # )
        # self.grad2 = nn.Sequential(
        #     BasicConv2d(in_channels, in_channels, kernel_size = (1, 5), padding = (0, 2)),
        #     BasicConv2d(in_channels, in_channels, kernel_size = (5, 1), padding = (2, 0)),
        #     BasicConv2d(in_channels, in_channels, kernel_size = 3, padding = 5, dilation = 5)
        # )
        # self.grad3 = nn.Sequential(
        #     BasicConv2d(in_channels, in_channels, kernel_size = (1, 7), padding = (0, 3)),
        #     BasicConv2d(in_channels, in_channels, kernel_size = (7, 1), padding = (3, 0)),
        #     BasicConv2d(in_channels, in_channels, kernel_size = 3, padding = 7, dilation = 7)
        # )
        # self.grad4 = nn.Sequential(
        #     BasicConv2d(in_channels, in_channels, kernel_size = (1, 9), padding = (0, 4)),
        #     BasicConv2d(in_channels, in_channels, kernel_size = (9, 1), padding = (4, 0)),
        #     BasicConv2d(in_channels, in_channels, kernel_size = 3, padding = 9, dilation = 9)
        # )
        # self.darkchannelModule = nn.BatchNorm2d(in_channels)
        # extra
        rdb_in_channels = depth_rate
        for i in range(height):
            for j in range(width - 1):
                self.rdb_module.update({'{}_{}'.format(i, j): RDB(rdb_in_channels, num_dense_layer, growth_rate)})
                DCL = int(2 ** i)
                self.dark_rdb_module.update({'{}_{}'.format(i, j): RDB(DCL, num_dense_layer, DCL)})
            rdb_in_channels *= stride

        _in_channels = depth_rate
        for i in range(height - 1):
            for j in range(width // 2):
                self.downsample_module.update({'{}_{}'.format(i, j): DownSample(_in_channels)})
                self.dark_downsample_module.update({'{}_{}'.format(i, j): DownSample(int(2 ** i))})
            _in_channels *= stride

        for i in range(height - 2, -1, -1):
            for j in range(width // 2, width):
                self.upsample_module.update({'{}_{}'.format(i, j): UpSample(_in_channels)})
                self.dark_upsample_module.update({'{}_{}'.format(i, j): UpSample(int(2 ** (i + 1)))})
            _in_channels //= stride
            
    def forward(self, x):
        # cur = 0
        
        # print("{} input: {}".format(cur, x.shape))
        # cur = cur + 1
        # x1 = self.grad1(x)
        # x2 = self.grad2(x)
        # x3 = self.grad3(x)
        # x4 = self.grad4(x)
        
        x_dc = GetDarkChannel(x, 15) 
        inp = self.conv_in(x)
        inp_dc = GetDarkChannel(inp, 15)
        inp_dc = inp_dc + self.dark_conv_in(x_dc)
        
        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        x_index_dc = [[0 for _ in range(self.width)] for _ in range(self.height)]
        i, j = 0, 0

        x_index[0][0] = self.rdb_in(inp)
        x_index_dc[0][0] = GetDarkChannel(x_index[0][0], 15)
        x_index_dc[0][0] = x_index_dc[0][0] + self.dark_rdb_in(inp_dc)
        
        for j in range(1, self.width // 2):
            x_index[0][j] = self.rdb_module['{}_{}'.format(0, j-1)](x_index[0][j-1])
            x_index_dc[0][j] = GetDarkChannel(x_index[0][j], 15)
            x_index_dc[0][j] = x_index_dc[0][j] + self.dark_rdb_module['{}_{}'.format(0, j-1)](x_index_dc[0][j-1])

        for i in range(1, self.height):
            x_index[i][0] = self.downsample_module['{}_{}'.format(i-1, 0)](x_index[i-1][0])
            x_index_dc[i][0] = GetDarkChannel(x_index[i][0], 15)
            x_index_dc[i][0] = x_index_dc[i][0] + self.dark_downsample_module['{}_{}'.format(i-1, 0)](x_index_dc[i-1][0])

        for i in range(1, self.height):
            for j in range(1, self.width // 2):
                channel_num = int(2**(i-1)*self.stride*self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.downsample_module['{}_{}'.format(i-1, j)](x_index[i-1][j])
                x_index_dc[i][j] = GetDarkChannel(x_index[i][j], 15)
                
                channel_num = int(2**i)
                x_index_dc[i][j] = self.dark_coefficient[i, j, 0, :channel_num][None, :, None, None] * self.dark_rdb_module['{}_{}'.format(i, j-1)](x_index_dc[i][j-1]) + \
                                self.dark_coefficient[i, j, 1, :channel_num][None, :, None, None] * self.dark_downsample_module['{}_{}'.format(i-1, j)](x_index_dc[i-1][j]) + \
                                x_index_dc[i][j]

        x_index[i][j+1] = self.rdb_module['{}_{}'.format(i, j)](x_index[i][j])
        x_index_dc[i][j + 1] = GetDarkChannel(x_index[i][j + 1], 15)
        x_index_dc[i][j+1] = 0.5 * x_index_dc[i][j + 1] + 0.5 * x_index_dc[i][j] + self.dark_rdb_module['{}_{}'.format(i, j)](x_index_dc[i][j])
        
        k = j

        for j in range(self.width // 2 + 1, self.width):
            x_index[i][j] = self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1])
            x_index_dc[i][j] = GetDarkChannel(x_index[i][j], 15)
            x_index_dc[i][j] = 0.5 * x_index_dc[i][j] + 0.5 * x_index_dc[i][5 - j] + self.dark_rdb_module['{}_{}'.format(i, j-1)](x_index_dc[i][j-1])

        for i in range(self.height - 2, -1, -1):
            channel_num = int(2 ** (i-1) * self.stride * self.depth_rate)
            x_index[i][k+1] = self.coefficient[i, k+1, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, k)](x_index[i][k]) + \
                              self.coefficient[i, k+1, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, k+1)](x_index[i+1][k+1], x_index[i][k].size())
            x_index_dc[i][k + 1] = GetDarkChannel(x_index[i][k + 1], 15)
            
            channel_num = int(2 ** i)
            x_index_dc[i][k+1] = self.dark_coefficient[i, k+1, 0, :channel_num][None, :, None, None] * self.dark_rdb_module['{}_{}'.format(i, k)](x_index_dc[i][k]) + \
                              self.dark_coefficient[i, k+1, 1, :channel_num][None, :, None, None] * self.dark_upsample_module['{}_{}'.format(i, k+1)](x_index_dc[i+1][k+1], x_index_dc[i][k].size()) + \
                              0.5 * x_index_dc[i][k + 1] + 0.5 * x_index_dc[i][5 - (k + 1)]

        for i in range(self.height - 2, -1, -1):
            for j in range(self.width // 2 + 1, self.width):
                channel_num = int(2 ** (i - 1) * self.stride * self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, j)](x_index[i+1][j], x_index[i][j-1].size())
                x_index_dc[i][j] = GetDarkChannel(x_index[i][j], 15)
                
                channel_num = int(2 ** i)
                x_index_dc[i][j] = self.dark_coefficient[i, j, 0, :channel_num][None, :, None, None] * self.dark_rdb_module['{}_{}'.format(i, j-1)](x_index_dc[i][j-1]) + \
                                self.dark_coefficient[i, j, 1, :channel_num][None, :, None, None] * self.dark_upsample_module['{}_{}'.format(i, j)](x_index_dc[i+1][j], x_index_dc[i][j-1].size()) + \
                                0.5 * x_index_dc[i][j] + 0.5 * x_index_dc[i][5 - j]

        out = self.rdb_out(x_index[i][j])
        # out = F.relu(self.conv_out(out))
        out_dc = GetDarkChannel(out, 15)
        
        out_dc = 0.5 * out_dc + 0.5 * x_dc + self.dark_rdb_out(x_index_dc[i][j])
        out = F.relu(self.conv_out(out))
        out = torch.cat((out, out_dc), dim = 1)
        out = F.relu(self.finalConv(out))
        
        return out, out_dc
    
class RandomGridDehazeNet(nn.Module):
    def __init__(self, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(RandomGridDehazeNet, self).__init__()
        self.rdb_module = nn.ModuleDict()
        self.upsample_module = nn.ModuleDict()
        self.downsample_module = nn.ModuleDict()
        self.height = height
        self.width = width
        self.stride = stride
        self.depth_rate = depth_rate
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((height, width, 2, depth_rate*stride**(height-1)))), requires_grad=attention)
        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb_out = RDB(depth_rate, num_dense_layer, growth_rate)

        rdb_in_channels = depth_rate
        for i in range(height):
            for j in range(width - 1):
                self.rdb_module.update({'{}_{}'.format(i, j): RDB(rdb_in_channels, num_dense_layer, growth_rate)})
            rdb_in_channels *= stride

        _in_channels = depth_rate
        for i in range(height - 1):
            for j in range(width // 2):
                self.downsample_module.update({'{}_{}'.format(i, j): DownSample(_in_channels)})
            _in_channels *= stride

        for i in range(height - 2, -1, -1):
            for j in range(width // 2, width):
                self.upsample_module.update({'{}_{}'.format(i, j): UpSample(_in_channels)})
            _in_channels //= stride

    def forward(self, x):
        cur = 0
        
        # print("{} input: {}".format(cur, x.shape))
        cur = cur + 1
        inp = self.conv_in(x)
        # print("{} conv_in: {}".format(cur, inp.shape))
        cur = cur + 1
        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        i, j = 0, 0

        x_index[0][0] = self.rdb_in(inp)
        # print("{} x[0][0]: {}".format(cur, x_index[0][0].shape))
        cur = cur + 1
        for j in range(1, self.width // 2):
            randomLayer = random.randint(0, j - 1)
            x_index[0][j] = self.rdb_module['{}_{}'.format(0, randomLayer)](x_index[0][randomLayer])
            # print("{}: x[{}][{}]{}".format(cur, 0, j, x_index[0][j].shape))
            cur = cur + 1

        for i in range(1, self.height):
            x_index[i][0] = self.downsample_module['{}_{}'.format(i-1, 0)](x_index[i-1][0])
            # print("{}: x[{}][{}]{}".format(cur, i, 0, x_index[i][0].shape))
            cur = cur + 1

        for i in range(1, self.height):
            for j in range(1, self.width // 2):
                randomLayer = random.randint(0, j - 1)
                channel_num = int(2**(i-1)*self.stride*self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, randomLayer)](x_index[i][randomLayer]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.downsample_module['{}_{}'.format(i-1, j)](x_index[i-1][j])
                # x_index[i][j] = self.rdb_module['{}_{}'.format(i, randomLayer)](x_index[i][randomLayer]) + self.downsample_module['{}_{}'.format(i-1, j)](x_index[i-1][j])
                # print("{}: x[{}][{}]{}".format(cur, i, j, x_index[i][j].shape))
                cur = cur + 1

        randomLayer = random.randint(0, j)
        x_index[i][j+1] = self.rdb_module['{}_{}'.format(i, randomLayer)](x_index[i][randomLayer])
        # print("{}: x[{}][{}]{}".format(cur, i, j + 1, x_index[i][j + 1].shape))
        cur = cur + 1
        k = j

        for j in range(self.width // 2 + 1, self.width):
            randomLayer = random.randint(0, j - 1)
            x_index[i][j] = self.rdb_module['{}_{}'.format(i, randomLayer)](x_index[i][randomLayer])
            # print("{}: x[{}][{}]{}".format(cur, i, j, x_index[i][j].shape))
            cur = cur + 1

        for i in range(self.height - 2, -1, -1):
            randomLayer = random.randint(0, k)
            channel_num = int(2 ** (i-1) * self.stride * self.depth_rate)
            x_index[i][k+1] = self.coefficient[i, k+1, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, randomLayer)](x_index[i][randomLayer]) + \
                              self.coefficient[i, k+1, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, k+1)](x_index[i+1][k+1], x_index[i][k].size())
            # x_index[i][k+1] = self.rdb_module['{}_{}'.format(i, randomLayer)](x_index[i][randomLayer]) + self.upsample_module['{}_{}'.format(i, k+1)](x_index[i+1][k+1], x_index[i][k].size())
            # print("{}: x[{}][{}]{}".format(cur, i, k + 1, x_index[i][k+1].shape))
            cur = cur + 1

        for i in range(self.height - 2, -1, -1):
            for j in range(self.width // 2 + 1, self.width):
                randomLayer = random.randint(0, j - 1)
                channel_num = int(2 ** (i - 1) * self.stride * self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, randomLayer)](x_index[i][randomLayer]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, j)](x_index[i+1][j], x_index[i][j-1].size())
                # x_index[i][j] = self.rdb_module['{}_{}'.format(i, randomLayer)](x_index[i][randomLayer]) + self.upsample_module['{}_{}'.format(i, j)](x_index[i+1][j], x_index[i][j-1].size())
                # print("{}: x[{}][{}]{}".format(cur, i, j, x_index[i][j].shape))
                cur = cur + 1

        out = self.rdb_out(x_index[i][j])
        # print("{}: x[{}][{}]{}".format(cur, i, j, out.shape))
        cur = cur + 1
        out = F.relu(self.conv_out(out))
        # print("{} out: {}".format(cur, out.shape))
        cur = cur + 1

        return out
        
class MyNet(nn.Module):
    def __init__(self, in_channels=3, depth_rate=8, kernel_size=3):
        super(MyNet, self).__init__()
        self.convin = nn.Conv2d(in_channels, depth_rate, kernel_size = 3, padding = (3 - 1) // 2)
        self.convout = nn.Conv2d(depth_rate, in_channels, kernel_size = 3, padding = (3 - 1) // 2)
        self.conv = nn.ModuleDict()
        for i in range(0, 4):
            for j in range(0, 6):
                if j == 0:
                    self.conv.update({'{}_{}'.format(i, j): nn.Conv2d(depth_rate, depth_rate * 2, kernel_size = 5, padding = (5 - 1) // 2)})
                if j == 1:
                    self.conv.update({'{}_{}'.format(i, j): nn.Conv2d(depth_rate * 2, depth_rate * 4, kernel_size = 7, padding = (7 - 1) // 2)})
                if j == 2:
                    self.conv.update({'{}_{}'.format(i, j): nn.Conv2d(depth_rate * 4, depth_rate * 8, kernel_size = 9, padding = (9 - 1) // 2)})
                if j == 3:
                    self.conv.update({'{}_{}'.format(i, j): nn.Conv2d(depth_rate * 8, depth_rate * 4, kernel_size = 9, padding = (9 - 1) // 2)})
                if j == 4:
                    self.conv.update({'{}_{}'.format(i, j): nn.Conv2d(depth_rate * 4, depth_rate * 2, kernel_size = 7, padding = (7 - 1) // 2)})
                if j == 5:
                    self.conv.update({'{}_{}'.format(i, j): nn.Conv2d(depth_rate * 2, depth_rate, kernel_size = 5, padding = (5 - 1) // 2)})

    def forward(self, x):
        out = self.convin(x)
        x_index = [[0 for _ in range(6)] for _ in range(4)]
        
        for i in range(0, 4):
            for j in range(0, 6):
                if j == 0:
                    randomLayer = random.randint(0, i)
                    if randomLayer == i:
                        x_index[i][j] = F.relu(self.conv['{}_{}'.format(i, j)](out))
                    else:
                        x_index[i][j] = F.relu(self.conv['{}_{}'.format(i, j)](x_index[randomLayer][5]))
                else:
                    randomLayer = random.randint(0, i)
                    x_index[i][j] = F.relu(self.conv['{}_{}'.format(i, j)](x_index[randomLayer][j - 1]))
            
        result = F.relu(self.convout(x_index[3][5]))

        return result