"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: train.py
about: main entrance for training the GridDehazeNet
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torchvision
import numpy
import cv2
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from train_data import TrainData
from val_data import ValData
from model import GridDehazeNet, RandomGridDehazeNet, MyNet
from utils import to_psnr, print_log, validation, adjust_learning_rate, RandomValidation
from torchvision.models import vgg16
from torchvision.transforms import Compose, ToTensor, Normalize
from perceptual import LossNetwork
plt.switch_backend('agg')


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-3, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[240, 240], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=18, type=int)
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-category', help='Set image category (indoor or outdoor?)', default='outdoor', type=str)
args = parser.parse_args()

writer = SummaryWriter()
learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
category = args.category
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\n'
      'num_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\ncategory: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, network_height, network_width, num_dense_layer, growth_rate, lambda_loss, category))

# --- Set category-specific hyper-parameters  --- #
if category == 'indoor':
    num_epochs = 100
    train_data_dir = './data/train/indoor/'
    val_data_dir = './data/test/SOTS/indoor/'
elif category == 'outdoor':
    num_epochs = 40
    train_data_dir = './data/train/outdoor/'
    val_data_dir = './data/test/SOTS/outdoor/'
else:
    raise Exception('Wrong image category. Set it to indoor or outdoor for RESIDE dateset.')

def GetGradImage_Tensor2Tensor(dehaze):
    unloader = torchvision.transforms.ToPILImage()
    ims = dehaze.clone()
    
    for i in range(0, ims.shape[0]):
        im = ims[i,:,:,:]
        im = im.squeeze(0)
        im = unloader(im)
        im = numpy.array(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.Laplacian(im, -1)
        im = cv2.equalizeHist(im)
        im = cv2.blur(im, (7, 7))
        im = Image.fromarray(im.astype('uint8'))
        transform_im = Compose([ToTensor(), Normalize((0.5), (0.5))])
        im = transform_im(im)
        im = im.unsqueeze(0)
        if i == 0:
            ans = im
        else:
            ans = torch.cat((ans, im), dim = 0)
    
    return ans

def GetDarkChannel(img, sz):
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

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
# net = MyNet()
net = GridDehazeNet(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
# preNet1 = GridDehazeNet(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
# preNet2 = GridDehazeNet(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)

# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)
# preNet1 = preNet1.to(device)
# preNet1 = nn.DataParallel(preNet1, device_ids=device_ids)
# preNet2 = preNet2.to(device)
# preNet2 = nn.DataParallel(preNet2, device_ids=device_ids)

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model)
loss_network.eval()


# --- Load the network weight --- #
try:
    net.load_state_dict(torch.load('./{}_haze_best_{}_{}'.format(category, network_height, network_width)))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')
# preNet1.load_state_dict(torch.load('./GradGridDehazeNet240/{}_haze_best_{}_{}'.format(category, network_height, network_width)))
# preNet1.eval()
# state2 = torch.load('./GradGridDehazeNet120/{}_haze_best_{}_{}'.format(category, network_height, network_width))
# preNet2.load_state_dict(state2['model'])
# preNet2.eval()

# --- Calculate all trainable parameters in network --- #
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))


# --- Load training data and validation/test data --- #
train_data_loader = DataLoader(TrainData(crop_size, train_data_dir), batch_size=train_batch_size, shuffle=True, num_workers=24)
val_data_loader = DataLoader(ValData(val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=24)


# --- Previous PSNR and SSIM in testing --- #
# old_val_psnr, old_val_ssim = validation(net, val_data_loader, device, category)
# print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))
old_val_psnr = 0
num = 0
for epoch in range(num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch, category=category)

    for batch_id, train_data in enumerate(train_data_loader):

        haze, gt = train_data
        haze = haze.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        # extra
        # dehaze = preNet1(haze)
        # dehaze = dehaze.to(device)
        # newGradDehaze = GetGradImage_Tensor2Tensor(dehaze)
        # newGradDehaze = newGradDehaze.to(device)
        
        # dehaze = torch.cat((dehaze, newGradDehaze), dim = 1)
        # dehaze = preNet2(dehaze)
        
        # newGradDehaze = GetGradImage_Tensor2Tensor(dehaze)
        # newGradDehaze = newGradDehaze.to(device)
        
        # dehaze = torch.cat((dehaze, newGradDehaze), dim = 1)
        dehaze, dc = net(haze)
        # extra
        smooth_loss = F.smooth_l1_loss(dehaze, gt)
        perceptual_loss = loss_network(dehaze, gt)
        dc_loss = F.smooth_l1_loss(dc, GetDarkChannel(gt, 15))
        
        loss = smooth_loss + lambda_loss*perceptual_loss + 0.25 * dc_loss
        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, gt))

        print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))
        writer.add_scalar('scalar/loss_w', loss, num)
        writer.add_scalar('scalar/psnr_w', sum(psnr_list) / len(psnr_list), num)
        num = num + 1

    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)

    # --- Save the network parameters --- #
    state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, '{}_haze_{}_{}'.format(category, network_height, network_width))

    # --- Use the evaluation model in testing --- #
    net.eval()

    val_psnr, val_ssim = validation(net, val_data_loader, device, category)
    one_epoch_time = time.time() - start_time
    print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category)

    # --- update the network weight --- #
    if epoch == 0 or val_psnr >= old_val_psnr:
        print("haze_best is {} epoch".format(epoch))
        state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, '{}_haze_best_{}_{}'.format(category, network_height, network_width))
        old_val_psnr = val_psnr
