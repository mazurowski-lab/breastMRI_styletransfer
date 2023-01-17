from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04
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
import pandas as pd
from data import func_identity, func_neg, func_log, func_gamma, func_piecewise_linear, func_sobelx, func_sobely, dicom_loader, make_dataset, func_gamma_neg, func_exp
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import write_2images
from random import sample
from scipy.stats import ttest_ind

from validation import Validator

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--csv_file', type=str, help="style code csv file path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--iterations', type=str, help="number of iterations when checkpoint saved")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--trainer', type=str, default='STAR', help="STAR")
parser.add_argument('--trans_func', type=str, help="image transformation function")
parser.add_argument('--paper_experiment', type=bool, default=False, help='are we running the inter-hospital scanner experiment from the paper?')
parser.add_argument('--most_rep_style_num_data', type=int, help='number of style codes to use to compute most rep style code from test B (for above experiment)')
parser.add_argument('--fix_test', action='store_true', help='fix 5 images to test on?')
parser.add_argument('--save_transferred_imgs', action='store_true')
parser.add_argument('--validate', action='store_true')
opts = parser.parse_args()

func_dict = {'func_neg': func_neg, 'func_identity': func_identity,  'func_log': func_log, 'func_gamma': func_gamma, 'func_piecewise_linear': func_piecewise_linear,
            'func_sobelx': func_sobelx, 'func_sobely': func_sobely, 'func_gamma_neg': func_gamma_neg, 'func_exp': func_exp}

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=dicom_loader, trans_func=None):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
        self.trans_func = trans_func

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)

        if self.trans_func:
            img = func_dict[self.trans_func](img)


        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


def get_data_loader_folder(input_folder, trans_func, batch_size, train, new_size, return_paths = False, num_workers=4, crop=False):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(new_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))])

    dataset = ImageFolder(input_folder, transform=transform, return_paths=return_paths, trans_func=trans_func)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
    return loader

def mean_abs_err(a, b):
  return np.mean(np.abs(a - b))


def mean_abs_err_torch(input, target):
    return torch.mean(torch.abs(input - target))


torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting                                                                                                                     
config = get_config(opts.config)
display_size = config['display_size']
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in config['device_ids'])

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
batch_size = 1
num_workers = config['num_workers']
if 'new_size' in config:
    new_size_a = new_size_b = config['new_size']
else:
    new_size_a = config['new_size_a']
    new_size_b = config['new_size_b']



test_loader_a = get_data_loader_folder(os.path.join(config['data_root'], 'testA'), None, batch_size, False,
                                                new_size_a, True, num_workers, False)

test_loader_aT = get_data_loader_folder(os.path.join(config['data_root'], 'testA'), opts.trans_func, batch_size, False,
                                                new_size_a, False, num_workers, False)


test_loader_b = get_data_loader_folder(os.path.join(config['data_root'], 'testB'), None, batch_size, False,
                                                new_size_a, False, num_workers, False)

test_loader_bT = get_data_loader_folder(os.path.join(config['data_root'], 'testB'), opts.trans_func, batch_size, False,
                                                new_size_a, False, num_workers, False)



trainer.cuda()
trainer.eval()
encode = trainer.gen.encode  # encode function                                                                                            
decode = trainer.gen.decode  # decode function                                                                                                

# Find the most representative style vector
df = pd.read_csv(opts.csv_file)

# if we want to compute most rep. style from a certain subset of the df:
if opts.paper_experiment:
    print('computing most rep. style code from {} codes originating from testB folder...'.format(opts.most_rep_style_num_data))
    n_subset = opts.most_rep_style_num_data

    # drop anything not from testA
    drop_range = list(range(0, 50, 2))
    df = df.drop(drop_range)
    # keep only a random subset
    keep_indices = sample(range(len(drop_range)), n_subset)

    df = df.iloc[keep_indices]

pos1 = np.array(df['position1'])
pos2 = np.array(df['position2'])
pos3 = np.array(df['position3'])
pos4 = np.array(df['position4'])
pos5 = np.array(df['position5'])
pos6 = np.array(df['position6'])
pos7 = np.array(df['position7'])
pos8 = np.array(df['position8'])

style_vector = np.array([pos1,pos2,pos3,pos4,pos5,pos6,pos7,pos8])
style_vector = np.transpose(style_vector)

dist = []
for i1, vec1 in enumerate(style_vector):
    vec1_dist = []
    for i2, vec2 in enumerate(style_vector):
        if i1 != i2:
            vec1_dist.append(mean_abs_err(vec1, vec2))
    dist.append(np.mean(vec1_dist))


# The most representative vector
style_repr = style_vector[np.argmin(dist)]
style_repr = torch.tensor(style_repr).float()


# MAEs between: (transformed, transferred), (g.t., transformed), (g.t., transferred)
total_err_transform_transferred = 0
total_err_gt_transform = 0
total_err_gt_transferred = 0

errs_transform_transferred = []
errs_gt_transform = []
errs_gt_transferred = []

# Set up validation
if opts.validate:
    validator = Validator(config)
    all_trans_errs = validator.validate_on_all_transformations(trainer)
    print('Validation : {}'.format(all_trans_errs))

# inference                                                                                                                                   
with torch.no_grad():
    fid_dir = 'imgs_for_fid'
    if opts.paper_experiment:
        s = Variable(style_repr.cuda()).unsqueeze(0)
        x_T_x = [] # reconstructed imgs to visualize
        x_aT = []
        x_a = []

        n_visualize = 5
        visualize_indices = sample(range(len(test_loader_a.dataset)), n_visualize)
        # ^ visualize reconstructions randomly chosen from test set A
        for batch_idx, (datapoint_a, datapath_a) in enumerate(test_loader_a):
            # load datapoint and its transformed counterpart
            datapoint_aT = test_loader_aT.dataset[batch_idx]
            # add batch dim to make shape correct
            datapoint_aT = datapoint_aT.unsqueeze(0)
            datapoint_aT = datapoint_aT.cuda() #g.t. that we want to match
            datapoint_a = datapoint_a.cuda()

            # autoencode transformed datapoint to nontransformed style and compare
            content, _ = encode(datapoint_a)
            transferred_img = decode(content, s)

            err_transform_transferred = mean_abs_err_torch(transferred_img, datapoint_aT)
            err_gt_transform = mean_abs_err_torch(datapoint_a, datapoint_aT)
            err_gt_transferred = mean_abs_err_torch(datapoint_a, transferred_img)

            total_err_transform_transferred += err_transform_transferred
            total_err_gt_transform += err_gt_transform
            total_err_gt_transferred += err_gt_transferred

            errs_gt_transform.append(err_gt_transform.item())
            errs_gt_transferred.append(err_gt_transferred.item())
            errs_transform_transferred.append(err_transform_transferred.item())

            if opts.save_transferred_imgs:
                vutils.save_image(datapoint_a, os.path.join(fid_dir, 'original', 'img_{}.png'.format(batch_idx)), normalize=True)
                vutils.save_image(transferred_img, os.path.join(fid_dir, 'transferred', 'img_{}.png'.format(batch_idx)), normalize=True)
                vutils.save_image(datapoint_aT, os.path.join(fid_dir, 'transformed', 'img_{}.png'.format(batch_idx)), normalize=True)

            # save for visualization
            if batch_idx in visualize_indices:
                print(datapath_a)
                x_T_x.append(transferred_img)
                x_aT.append(datapoint_aT)
                x_a.append(datapoint_a)
                #vutils.save_image(datapoint_a, 'img_test_xa.png')


        total_err_transform_transferred = total_err_transform_transferred/len(test_loader_a.dataset)
        total_err_gt_transform = total_err_gt_transform/len(test_loader_a.dataset)
        total_err_gt_transferred = total_err_gt_transferred/len(test_loader_a.dataset)
        x_T_x = torch.cat(x_T_x)
        x_a = torch.cat(x_a)
        x_aT = torch.cat(x_aT)
        print(x_a.shape)
        write_2images([x_a,x_aT,x_T_x], display_size, opts.output_folder, 'test_hospital_sim_%08d_%s_{}'.format(opts.most_rep_style_num_data) % (int(opts.iterations), opts.trans_func), "")
        #outputs = (outputs + 1) / 2.                                                                                                              
        print('MAE(transformed, transferred) = {}'.format(total_err_transform_transferred.item()))
        print('MAE(gt, transformed) = {}'.format(total_err_gt_transform.item()))
        print('MAE(gt, transferred) = {}'.format(total_err_gt_transferred.item()))


        # statistical testing
        t_statistic, pval = ttest_ind(errs_gt_transform, errs_gt_transferred)
        print('stat testing for MAE(gt, transformed) vs. MAE(gt, transferred):')
        print('t_statistic = {}, pval = {}'.format(t_statistic, pval))

        # quick code for saving vertical column for figures
        test_grid = vutils.make_grid(x_a, nrow=1, padding=0, normalize=True)
        vutils.save_image(test_grid, 'vert_grid_in.png')

        test_grid = vutils.make_grid(x_T_x, nrow=1, padding=0, normalize=True)
        vutils.save_image(test_grid, 'vert_grid_out.png')
    else:
        s = Variable(style_repr.cuda()).unsqueeze(0)
        x_T_x = [] # reconstructed imgs to visualize
        x_aT = []
        x_a = []

        n_visualize = 5
        if opts.fix_test:
            visualize_indices = range(n_visualize)
 
        else:
            visualize_indices = sample(range(len(test_loader_a.dataset)), n_visualize)
        # ^ visualize reconstructions randomly chosen from test set A
        for batch_idx, datapoint_a in enumerate(test_loader_a):
            # load datapoint and its transformed counterpart
            datapoint_aT = test_loader_aT.dataset[batch_idx]
            # add batch dim to make shape correct
            datapoint_aT = datapoint_aT.unsqueeze(0)
            datapoint_aT = datapoint_aT.cuda()
            datapoint_a = datapoint_a.cuda()

            # autoencode transformed datapoint to nontransformed style and compare
            content, _ = encode(datapoint_aT)
            transferred_img = decode(content, s)

            err_transform_transferred = mean_abs_err_torch(transferred_img, datapoint_aT)
            err_gt_transform = mean_abs_err_torch(datapoint_a, datapoint_aT)
            err_gt_transferred = mean_abs_err_torch(datapoint_a, transferred_img)

            total_err_transform_transferred += err_transform_transferred
            total_err_gt_transform += err_gt_transform
            total_err_gt_transferred += err_gt_transferred

            # save for visualization
            if batch_idx in visualize_indices:
                x_T_x.append(transferred_img)
                x_aT.append(datapoint_aT)
                x_a.append(datapoint_a)


        for batch_idx, datapoint_b in enumerate(test_loader_b):
            # load datapoint and its transformed counterpart
            datapoint_bT = test_loader_bT.dataset[batch_idx]
            # add batch dim to make shape correct
            datapoint_bT = datapoint_bT.unsqueeze(0)
            datapoint_bT = datapoint_bT.cuda()
            datapoint_b = datapoint_b.cuda()

            # autoencode transformed datapoint to nontransformed style and compare
            content, _ = encode(datapoint_bT)
            transferred_img = decode(content, s)

            err_transform_transferred = mean_abs_err_torch(transferred_img, datapoint_bT)
            err_gt_transform = mean_abs_err_torch(datapoint_b, datapoint_bT)
            err_gt_transferred = mean_abs_err_torch(datapoint_b, transferred_img)

            total_err_transform_transferred += err_transform_transferred
            total_err_gt_transform += err_gt_transform
            total_err_gt_transferred += err_gt_transferred


        total_err_transform_transferred = total_err_transform_transferred/(len(test_loader_a.dataset) + len(test_loader_b.dataset))
        x_T_x = torch.cat(x_T_x)
        x_a = torch.cat(x_a)
        x_aT = torch.cat(x_aT)
        write_2images([x_a,x_aT,x_T_x], display_size, opts.output_folder, 'test_%08d_%s' % (int(opts.iterations), opts.trans_func), "")

        print('MAE = {}'.format(total_err_transform_transferred.item()))
