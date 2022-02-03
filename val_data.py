"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: val_data.py
about: build the validation/test dataset
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from glob import glob
import os

# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        val_list = val_data_dir + 'val_list.txt'
        with open(val_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('_')[0] + '.png' for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_img = Image.open(self.val_data_dir + 'hazy/' + haze_name)
        gt_img = Image.open(self.val_data_dir + 'clear/' + gt_name)
        
        # extra
        # grad_img = Image.open(self.val_data_dir + 'Grad/Grad_' + haze_name[:-3] + 'jpg')
        # r, g, b = haze_img.split()
        # r1 = grad_img.split()
        # exhaze_img = Image.merge("RGBA", (r, g, b, grad_img))
        # exhaze_img = np.zeros((haze_img.shape[0], haze_img.shape[1], haze_img.shape[2] + 1), dtype = haze_img.dtype)
        # exhaze_img[:,:,0] = haze_img[:,:,0]
        # exhaze_img[:,:,1] = haze_img[:,:,1]
        # exhaze_img[:,:,2] = haze_img[:,:,2]
        # exhaze_img[:,:,3] = grad_img[:,:,0]
        # extra
        
        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        # extra
        haze = transform_haze(haze_img)
        # haze = transform_haze(exhaze_img)
        # extra
        gt = transform_gt(gt_img)

        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)

class TestRealData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        fpaths = glob(os.path.join(val_data_dir, '*.png')) + glob(os.path.join(val_data_dir, '*.jpeg')) + glob(os.path.join(val_data_dir, '*.jpg'))
        haze_names = []
        for path in fpaths:
            if path.split('/')[-1].startswith('DC') == True:
                continue
            if path.split('/')[-1].startswith('guide') == True:
                continue
            if path.split('/')[-1].startswith('Grad') == True:
                continue
            haze_names.append(path.split('/')[-1])
            print(path.split('/')[-1])
        self.haze_names = haze_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        haze_name = self.haze_names[index]
        haze_img = Image.open(os.path.join(self.val_data_dir, haze_name))
        
        # extra
        # if haze_name.endswith('jpg'):
        #     grad_img = Image.open(self.val_data_dir + 'Grad_' + haze_name[:-3] + 'jpg')
        # if haze_name.endswith('jpeg'):
        #     grad_img = Image.open(self.val_data_dir + 'Grad_' + haze_name[:-4] + 'jpg')
        # if haze_name.endswith('png'):
        #     grad_img = Image.open(self.val_data_dir + 'Grad_' + haze_name[:-3] + 'jpg')
        # r, g, b = haze_img.split()
        # r1 = grad_img.split()
        # exhaze_img = Image.merge("RGBA", (r, g, b, grad_img))
        # extra
        
        
        # --- Transform to tensor --- #
        # extra
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        haze = transform_haze(haze_img)
        # extra
        return haze, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)