"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.utils.data as data
import os.path
from pydicom import dcmread
import numpy as np
import cv2
import random
from functools import partial

from transforms import *
# TRANSFORMATION FUNCTION DICTS
# (all transformations; seen in training or otherwise)
# fixed transformation functions
func_dict_default = {'func_neg': func_neg, 'func_log': func_log, 'func_gamma': func_gamma, 'func_piecewise_linear': func_piecewise_linear, 'func_sobelx': func_sobelx, 'func_sobely': func_sobely, 'func_identity': func_identity, 'func_gamma_neg': func_gamma_neg, 'func_exp': func_exp}
# transformation functions whose parameters are randomly selected at each call
func_dict_randomized = {'func_neg': func_neg_randomized, 'func_log': func_log_randomized, 'func_gamma': func_gamma_randomized, 'func_piecewise_linear': func_piecewise_linear_randomized,  'func_sobelx': func_sobelx_randomized, 'func_sobely': func_sobely_randomized, 'func_identity': func_identity_randomized}


# transformation funcs to randomly choose from for training
func_list = [func_neg, func_piecewise_linear, func_sobelx, func_sobely, func_identity]

func_name = ["func_neg", "func_piecewise_linear", "func_sobelx", "func_sobely", "func_identity"]


def default_loader(path):
    return Image.open(path).convert('RGB')

def dicom_loader(path):
    obj = dcmread(path, force=True)
    if obj.PixelSpacing[0] != obj.PixelSpacing[1]:
        raise UserWarning("Diffrent spacing {} ".format(obj.PixelSpacing))
    img = obj.pixel_array
    img_type = obj.PhotometricInterpretation

    # uint16 -> float
    img = img.astype(np.float) * 255. / img.max()
    # float -> unit8
    img = img.astype(np.uint8)
    if img_type == "MONOCHROME1":
        img = np.invert(img)
    return img
    

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=dicom_loader, trans_func=None):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader
        self.trans_func = trans_func

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.trans_func is not None:
            func = random.choice(func_list)
            img = func(img)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

#IMG_EXTENSIONS = [
#    '.jpg', '.JPG', '.jpeg', '.JPEG',
#    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
#]

IMG_EXTENSIONS = ['.dcm']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


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
            func = random.choice(func_list)
            img = func(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class ImageFolder_Double(data.Dataset):
    def __init__(self, root1, root2, transform=None, return_paths=False,
                 loader=dicom_loader, trans_func=None, specific_pretransform=None, continuize_transforms=False):
        imgs1 = sorted(make_dataset(root1))
        imgs2 = sorted(make_dataset(root2))

        if len(imgs1) == 0:
            raise(RuntimeError("Found 0 images in: " + root1 + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        if len(imgs2) == 0:
            raise(RuntimeError("Found 0 images in: " + root2 + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        self.root1 = root1
        self.root2 = root2
        self.imgs1 = imgs1
        self.imgs2 = imgs2
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
        self.trans_func = trans_func
        self.specific_pretransform = specific_pretransform
        self.func_dict = func_dict_randomized if continuize_transforms else func_dict_default
        if continuize_transforms: print('using image transform functions with continuously random parameters.')
        self.continuize_transforms = continuize_transforms

    def __getitem__(self, index):
        path1 = self.imgs1[index]
        path2 = self.imgs2[index]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        # choose transformation function to apply to images
        func = random.choice(func_name)
        if self.specific_pretransform:
            func = self.specific_pretransform
            #print('applying {} transform to all...'.format(func))


        transformation = self.func_dict[func]
        if self.continuize_transforms:
            fixed_seed = np.random.randint(1000000)
            # this ^ function by default takes args of the img and of the seed
            # if func_dict_randomized is being used rather than func_dict_default
            # the seed needs to be fixed both time the fn is called
            transformation = partial(transformation, seed=fixed_seed) 
            
        if self.trans_func:
            img1 = transformation(img1)
            img2 = transformation(img2)
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.return_paths:
            return [img1, path1, img2, path2]
        else:
            return [img1, img2]

    def __len__(self):
        return len(self.imgs1)
