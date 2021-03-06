# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from val_data import ValData, TestRealData
from model import GridDehazeNet, MyNet
from utils import validation, RealValidation

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-category', help='Set image category (indoor or outdoor?)', default='outdoor', type=str)
args = parser.parse_args()

network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
category = args.category
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'

print('--- Hyper-parameters for testing ---')
print('val_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\nnum_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\ncategory: {}'
      .format(val_batch_size, network_height, network_width, num_dense_layer, growth_rate, lambda_loss, category))

# --- Set category-specific hyper-parameters  --- #
if category == 'indoor':
    val_data_dir = './data/test/SOTS/indoor/'
elif category == 'outdoor':
    val_data_dir = './data/test/SOTS/outdoor/hazy'
else:
    raise Exception('Wrong image category. Set it to indoor or outdoor for RESIDE dateset.')


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Validation data loader --- #
val_data_loader = DataLoader(TestRealData(val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=24)


# --- Define the network --- #
net = GridDehazeNet(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
net2 = GridDehazeNet(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
# net3 = GridDehazeNet(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
# net = MyNet()
# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)
# net2 = net2.to(device)
# net2 = nn.DataParallel(net2, device_ids=device_ids)
# net3 = net3.to(device)
# net3 = nn.DataParallel(net3, device_ids=device_ids)

# --- Load the network weight --- #
NetState = torch.load('./{}_haze_best_{}_{}'.format(category, network_height, network_width))
net.load_state_dict(NetState['model'])
print("loaded net1")
# NetState = torch.load('./GradGridDehazeNet120/{}_haze_best_{}_{}'.format(category, network_height, network_width))
# net2.load_state_dict(NetState['model'])
# print("loaded net2")
# NetState = torch.load('./GradGridDehazeNet60/{}_haze_best_{}_{}'.format(category, network_height, network_width))
# net3.load_state_dict(NetState['model'])


# --- Use the evaluation model in testing --- #
net.eval()
print('--- Testing starts! ---')
start_time = time.time()
RealValidation(net, val_data_loader, device, category, save_tag=True)
end_time = time.time() - start_time
print('validation time is {0:.4f}'.format(end_time))