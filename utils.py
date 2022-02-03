"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: utils.py
about: all utilities
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
import numpy
import cv2
import torchvision
from math import log10
from skimage import measure
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

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

def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [measure.compare_ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]

    return ssim_list


def validation(net, val_data_loader, device, category, save_tag=False):
    """
    :param net: GateDehazeNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: indoor or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            haze, gt, image_name = val_data
            haze = haze.to(device)
            gt = gt.to(device)
            # extra
            # dehaze = preNet1(haze)
            # newGradDehaze = GetGradImage_Tensor2Tensor(dehaze)
            # newGradDehaze = newGradDehaze.to(device)
        
            # dehaze = torch.cat((dehaze, newGradDehaze), dim = 1)
            # dehaze = preNet2(dehaze)
            # newGradDehaze = GetGradImage_Tensor2Tensor(dehaze)
            # newGradDehaze = newGradDehaze.to(device)
        
            # dehaze = torch.cat((dehaze, newGradDehaze), dim = 1)
            dehaze, dc = net(haze)
            
            # extra


        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(dehaze, gt))

        # --- Save image --- #
        if save_tag:
            save_image(dehaze, image_name, category)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim

def RandomValidation(net, val_data_loader, device, category, save_tag=False):
    """
    :param net: GateDehazeNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: indoor or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):
        
        with torch.no_grad():
            haze, gt, image_name = val_data
            averageDehaze = torch.zeros(haze.shape)
            averageDehaze = averageDehaze.to(device)
            haze = haze.to(device)
            gt = gt.to(device)
            for i in range(0, 5):
                dehaze = net(haze)
                averageDehaze = averageDehaze + dehaze
            
            averageDehaze = averageDehaze / 5

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(averageDehaze, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(averageDehaze, gt))

        # --- Save image --- #
        if save_tag:
            save_image(dehaze, image_name, category)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim

def RealValidation(net, net2, net3, val_data_loader, device, category, save_tag=False):
    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad():
            haze, image_name = val_data
            haze = haze.to(device)
            dehaze = net(haze)
            print("net1 finish")
            grad = GetGradImage_Tensor2Tensor(dehaze)
            grad = grad.to(device)
            dehaze = torch.cat((dehaze, grad), dim = 1)
            dehaze = net2(dehaze)
            print("net2 finish")
            # grad = GetGradImage_Tensor2Tensor(dehaze)
            # grad = grad.to(device)
            # dehaze = torch.cat((dehaze, grad), dim = 1)
            # dehaze = net3(dehaze)
            # print("net3 finish")
            
        fh_save_image(dehaze, image_name)


def fh_save_image(dehaze, image_name):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)

    for ind in range(batch_num):
        if image_name[ind].endswith('png'):   
            utils.save_image(dehaze_images[ind], './fh_outdoor_results/{}'.format(image_name[ind][:-3] + 'png'))
        if image_name[ind].endswith('jpeg'):   
            utils.save_image(dehaze_images[ind], './fh_outdoor_results/{}'.format(image_name[ind][:-4] + 'jpeg'))
        if image_name[ind].endswith('jpg'):   
            utils.save_image(dehaze_images[ind], './fh_outdoor_results/{}'.format(image_name[ind][:-3] + 'jpg'))

def save_image(dehaze, image_name, category):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)

    for ind in range(batch_num):
        utils.save_image(dehaze_images[ind], './{}_results/{}'.format(category, image_name[ind][:-3] + 'png'))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # --- Write the training log --- #
    with open('./training_log/{}_log.txt'.format(category), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)


def adjust_learning_rate(optimizer, epoch, category, lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 20 if category == 'indoor' else 8

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))
