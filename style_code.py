"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
# Used to get style codes for a given dataset
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04, get_data_loader_folder_double
from trainer import STAR_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import csv
from pydicom import dcmread

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--trainer', type=str, default='STAR', help="STAR")
parser.add_argument('--iterations', type=str, help="the number of iterations when checkpoint saved")
parser.add_argument('--dataset', type=str, default='train', help='get style codes for train or test set?')
parser.add_argument('--transform', type=str, help='transformation applied to data before extracting style')
parser.add_argument('--reverse_datasets', action='store_true', help='reverse the test datasets?')
opts = parser.parse_args()


def dicom_info(path):
    obj = dcmread(path, force=True)
    return obj.Manufacturer, obj.ManufacturerModelName

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)

# Setup model 
if opts.trainer == 'STAR':
    style_dim = config['gen']['style_dim']
    trainer = STAR_Trainer(config)
else:
    sys.exit("Only support STAR")

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen.load_state_dict(state_dict['a'])
    print('Successfully load model.')
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    trainer.gen.load_state_dict(state_dict['a'])
    print('Successfully load model.')

# Setup dataloader
batch_size = 1 #config['batch_size']
num_workers = config['num_workers']
if 'new_size' in config:
    new_size_a = new_size_b = config['new_size']
else:
    new_size_a = config['new_size_a']
    new_size_b = config['new_size_b']

whichdataset = opts.dataset
print('Extracting style codes for {} set...'.format(whichdataset))

whichtransform = opts.transform if opts.transform else False
if whichtransform:
    print('first applying {} transformation to data...'.format(whichtransform))

dataset1 = '{}A'.format(whichdataset)
dataset2 = '{}B'.format(whichdataset)
if opts.reverse_datasets:
    dataset1_temp = dataset1
    dataset2_temp = dataset2
    dataset2 = dataset1_temp
    dataset1 = dataset2_temp

# for use with test2.py: dataset1 is dataset of target style imgs, dataset 2 are imgs that are transferred to this target style

train_loader_ab = get_data_loader_folder_double(os.path.join(config['data_root'], dataset1), os.path.join(config['data_root'], dataset2), True, batch_size, True, new_size_a, True, num_workers, False, specific_pretransform=whichtransform)

trainer.cuda()
trainer.eval()
encode = trainer.gen.encode  # encode function
decode = trainer.gen.decode  # decode function

filename = "style_codes_{}_{}.csv".format(whichdataset, opts.iterations)
if whichtransform:
    filename = "style_codes_{}_{}_{}.csv".format(whichdataset, opts.iterations, whichtransform)

with torch.no_grad():
    with open(filename, "w", newline='') as file:
        writer = csv.writer(file) 
        writer.writerow(['path', 'manufacturer','manufacturer_model','position1','position2','position3','position4','position5','position6','position7','position8'])
        for it, (images_a, path_a, images_b, path_b) in enumerate(train_loader_ab):
            #print("Iteration: %08d" % (it + 1))
            # vutils.save_image(images_a, 'img_test_a_{}.png'.format(it), normalize=True)
            images_a = images_a.cuda()
            images_b = images_b.cuda()
            # Get Manufacture informattion 
            manufc_a, manufc_model_a =  dicom_info(path_a[0])
            manufc_b, manufc_model_b =  dicom_info(path_b[0])
            # Extract style code

            style_code_a = encode(images_a)[1]
            style_code_a_np = style_code_a.reshape((8)).cpu().numpy()
            writer.writerow([path_a[0],manufc_a,manufc_model_a,style_code_a_np[0],style_code_a_np[1],style_code_a_np[2],style_code_a_np[3],style_code_a_np[4],style_code_a_np[5],style_code_a_np[6],style_code_a_np[7]])

            style_code_b = encode(images_b)[1]
            style_code_b_np = style_code_b.reshape((8)).cpu().numpy()
            writer.writerow([path_b[0],manufc_b,manufc_model_b,style_code_b_np[0],style_code_b_np[1],style_code_b_np[2],style_code_b_np[3],style_code_b_np[4],style_code_b_np[5],style_code_b_np[6],style_code_b_np[7]])






