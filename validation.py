'''
Function(s) for computing validation score at some training iteration of this particular model
'''
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.utils.data as data
from torch.autograd import Variable
import os
import numpy as np

from data import dicom_loader, make_dataset, ImageFolder, ImageFolder_Double
from utils import get_data_loader_folder_double, get_data_loader_folder
# import different img transformation functions
from data import func_list, func_neg, func_log, func_gamma, func_piecewise_linear, func_sobelx, func_sobely, func_identity, func_exp

# import model stuff
from trainer import STAR_Trainer

# evaluation utils from testing file (test2.py)
# note: can't import from test2.py because that runs stuff in test2.py that I don't need run
func_dict = {'func_neg': func_neg, 'func_log': func_log, 'func_gamma': func_gamma, 'func_piecewise_linear': func_piecewise_linear,
        'func_sobelx': func_sobelx, 'func_sobely': func_sobely, 'func_identity': func_identity, 'func_exp': func_exp}

def mean_abs_err(a, b):
  return np.mean(np.abs(a - b))

def mean_abs_err_torch(input, target):
    return torch.mean(torch.abs(input - target))

# core validation stuff
class Validator:
    '''
    This class will keep validation data loaded to be evaluated on a given model;
    To be used through the training process
    '''
    def __init__(self, config):
        self.config = config
        # determine which transformations to validate on (not the ones used in training)
        possible_transform_funcs = [func_identity, func_neg, func_log, func_gamma, func_piecewise_linear, func_sobelx, func_sobely]
        self.nontraining_transform_funcs = [func for func in possible_transform_funcs if func not in func_list]

        # set up for each validation call

        # Load experiment setting                                                                                                                     
        display_size = config['display_size']
        
        ## Setup model                                                                                                                                  
        #self.trainer = AUTOENCODER_Trainer(config)

        # Setup dataloader                                                                                                                             
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        if 'new_size' in config:
            self.new_size_a = self.new_size_b = config['new_size']
        else:
            self.new_size_a = config['new_size_a']
            self.new_size_b = config['new_size_b']
    
        # set up non-transformed dataset
        self.validation_loader_a = get_data_loader_folder(os.path.join(self.config['data_root'], 'validationA'), None, self.batch_size, False,
                                                        self.new_size_a, False, self.num_workers, False)

        self.validation_loader_b = get_data_loader_folder(os.path.join(self.config['data_root'], 'validationB'), None, self.batch_size, False,
                                                        self.new_size_a, False, self.num_workers, False)

        # Setup dataloader for computing most rep. style code
        self.style_repr = None

        batch_size_stylecode = 1
        self.train_loader_ab = get_data_loader_folder_double(os.path.join(config['data_root'], 'trainA'), os.path.join(config['data_root'], 'trainB'),
                                                       False, batch_size_stylecode, True, self.new_size_a, True, self.num_workers, False)


    def compute_mostrep_stylecode(self, trainer):
        '''
        Compute most representative style code given current model iteration
        Saved as self.style_repr
        '''
        trainer.cuda()
        trainer.eval()
        encode = trainer.gen.encode  # encode function
        decode = trainer.gen.decode  # decode function

        all_style_codes = []

        with torch.no_grad():
            for it, (images_a, path_a, images_b, path_b) in enumerate(self.train_loader_ab):
                # note: confirmed that appending of codes from paths a, b, a, b ... matches
                # style_code.py
                images_a = images_a.cuda()
                images_b = images_b.cuda()
                # Extract style code
                style_code_a = encode(images_a)[1]
                style_code_a_np = style_code_a.reshape((8)).cpu().numpy()
                all_style_codes.append(style_code_a_np)

                style_code_b = encode(images_b)[1]
                style_code_b_np = style_code_b.reshape((8)).cpu().numpy()
                all_style_codes.append(style_code_b_np)

        # Find the most representative style vector
        style_vector = np.array(all_style_codes)

        #print('all_style_codes.shape = {}'.format(style_vector.shape))

        dist = []
        for i1, vec1 in enumerate(style_vector):
            vec1_dist = []
            for i2, vec2 in enumerate(style_vector):
                if i1 != i2:
                    vec1_dist.append(mean_abs_err(vec1, vec2))
            dist.append(np.mean(vec1_dist))


        # The most representative vector
        style_repr = style_vector[np.argmin(dist)]
        self.style_repr = torch.tensor(style_repr).float()
        #print(self.style_repr.shape)

    def validate_on_transformation(self, trainer, trans_func_name):
        '''
        perform validation by reconstructing on a given transformation; compute average error for the current model iteration
        '''

        ## load model from current iteration
        #try: # the state dict should be of the form gen_*.pt
        #   self.trainer.gen.load_state_dict(model_state_dict['a'])
        #except:
        #   self.trainer.gen.load_state_dict(model_state_dict['a'])

        # set up transformed dataset for the given transformation
        validation_loader_aT = get_data_loader_folder(os.path.join(self.config['data_root'], 'validationA'), trans_func_name, self.batch_size, False,
                                                        self.new_size_a, False, self.num_workers, False)

        validation_loader_bT = get_data_loader_folder(os.path.join(self.config['data_root'], 'validationB'), trans_func_name, self.batch_size, False,
                                                        self.new_size_a, False, self.num_workers, False)
        # test on all of validation set

        trainer.cuda()
        trainer.eval()
        encode = trainer.gen.encode  # encode function                                                                                            
        decode = trainer.gen.decode  # decode function                                                                                                

        total_err = 0

        with torch.no_grad():
            # most rep. style code
            s = Variable(self.style_repr.cuda()).unsqueeze(0)
            for batch_idx, datapoint_a in enumerate(self.validation_loader_a):
                # load datapoint and its transformed counterpart
                datapoint_aT = validation_loader_aT.dataset[batch_idx]
                # add batch dim to make shape correct
                datapoint_aT = datapoint_aT.unsqueeze(0)
                datapoint_aT = datapoint_aT.cuda()
                datapoint_a = datapoint_a.cuda()

                # autoencode transformed datapoint to nontransformed style and compare
                content, _ = encode(datapoint_aT)
                transferred_img = decode(content, s)
                err = mean_abs_err_torch(transferred_img, datapoint_a)
                total_err += err

            for batch_idx, datapoint_b in enumerate(self.validation_loader_b):
                # load datapoint and its transformed counterpart
                datapoint_bT = validation_loader_bT.dataset[batch_idx]
                # add batch dim to make shape correct
                datapoint_bT = datapoint_bT.unsqueeze(0)
                datapoint_bT = datapoint_bT.cuda()
                datapoint_b = datapoint_b.cuda()

                # autoencode transformed datapoint to nontransformed style and compare
                content, _ = encode(datapoint_bT)
                transferred_img = decode(content, s)
                err = mean_abs_err_torch(transferred_img, datapoint_b)
                total_err += err

            total_err = total_err/(len(self.validation_loader_a.dataset) + len(self.validation_loader_b.dataset))

        return total_err


    def validate_on_all_transformations(self, trainer):
        '''
        Returns dict with key of transformation name and value of the average error for that transform
        '''
        self.compute_mostrep_stylecode(trainer)

        all_trans_errs = {}
        for trans_func in self.nontraining_transform_funcs: 
            # get transformation func name from func itself
            trans_func_name = list(func_dict.keys())[list(func_dict.values()).index(trans_func)]
            assert type(trans_func_name) == str, 'type actually {}'.format(type(trans_func_name))

            trans_err = self.validate_on_transformation(trainer, trans_func_name)
            all_trans_errs[trans_func_name] = trans_err

        return all_trans_errs   
            
    def write_validation_to_log(self, summarywriter, all_trans_errs, iteration):
        '''
        Writes validation data for a given iteration to the 
        Tensorboard summary logged during training
        '''
        for trans_func_name in list(all_trans_errs.keys()):
            trans_err = all_trans_errs[trans_func_name]
            summarywriter.add_scalar('val_MAE_{}'.format(trans_func_name), trans_err, iteration + 1)
            
            
