"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer

# validation
from validation import Validator

import argparse
from torch.autograd import Variable
from trainer import STAR_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='STAR', help="STAR")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in config['device_ids'])

# Setup model and data loader
if opts.trainer == 'STAR':
    trainer = STAR_Trainer(config)
else:
    sys.exit("Only support STAR")

if config['train_parallel']:
    trainer = torch.nn.parallel.DataParallel(trainer, device_ids=config['device_ids'])
    trainer = trainer.module


trainer.cuda()

train_loader_ab, train_loader_aTbT, test_loader_ab, test_loader_aTbT = get_all_data_loaders(config)

train_display_images_ab = [train_loader_ab.dataset[i] for i in range(display_size)]
train_display_images_a = torch.stack([train_display_images_ab[i][0] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack([train_display_images_ab[i][1] for i in range(display_size)]).cuda()

train_display_images_aTbT = [train_loader_aTbT.dataset[i] for i in range(display_size)]
train_display_images_aT = torch.stack([train_display_images_aTbT[i][0] for i in range(display_size)]).cuda()
train_display_images_bT = torch.stack([train_display_images_aTbT[i][1] for i in range(display_size)]).cuda()

test_display_images_ab = [test_loader_ab.dataset[i] for i in range(display_size)]
test_display_images_a = torch.stack([test_display_images_ab[i][0] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack([test_display_images_ab[i][1] for i in range(display_size)]).cuda()

test_display_images_aTbT = [test_loader_aTbT.dataset[i] for i in range(display_size)]
test_display_images_aT = torch.stack([test_display_images_aTbT[i][0] for i in range(display_size)]).cuda()
test_display_images_bT = torch.stack([test_display_images_aTbT[i][1] for i in range(display_size)]).cuda()


# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0] + '_' + config['trans_func']
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Set up validation
validator = Validator(config)

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, ((images_a,images_b),(images_a_T,images_b_T)) in enumerate(zip(train_loader_ab, train_loader_aTbT)):
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        images_a_T, images_b_T = images_a_T.cuda().detach(), images_b_T.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.gen_update(images_a, images_a_T, images_b, images_b_T, config)
            trainer.update_learning_rate()
            torch.cuda.synchronize()

        # do validation
        all_trans_errs = validator.validate_on_all_transformations(trainer)
        print('Validation at iteration {}: {}'.format(iterations, all_trans_errs))
        # log validation
        validator.write_validation_to_log(train_writer, all_trans_errs, iterations)

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample_test(test_display_images_a, test_display_images_aT, test_display_images_b,  test_display_images_bT)
                train_image_outputs_a = trainer.sample(train_display_images_a, train_display_images_aT)
                train_image_outputs_b = trainer.sample(train_display_images_b, train_display_images_bT)
		

            write_2images([train_display_images_a, train_display_images_aT, train_display_images_b, train_display_images_bT], display_size, image_directory, 'origin_%08d' % (iterations + 1), "")
            write_2images(train_image_outputs_a, display_size, image_directory, 'train_%08d' % (iterations + 1), 'a')
            write_2images(train_image_outputs_b, display_size, image_directory, 'train_%08d' % (iterations + 1), 'b')
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1), '')

            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs_a = trainer.sample(train_display_images_a, train_display_images_aT)
                image_outputs_b = trainer.sample(train_display_images_b, train_display_images_bT)
            write_2images(image_outputs_a, display_size, image_directory, 'train_current','a')
            write_2images(image_outputs_b, display_size, image_directory, 'train_current','b') 


        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

