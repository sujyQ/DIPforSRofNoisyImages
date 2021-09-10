import logging
import time
import numpy as np
import os
import shutil
from collections import OrderedDict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
# from tensorboardX import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio
from collections import namedtuple
from skimage import io

from configs import getConfigs
from utils.common_utils import *
from utils.loss import VGG19_ContentLoss
from models import *
from models.downsampler import Downsampler
from models.discriminator import UNetD, Discriminator, WDiscriminator
# from torchsummary import summary

DenoizeResult = namedtuple("DenoizeResult", ['learned_LR', 'learned_HR', 'n', 'psnr', 'step'])

class Solver(object) :
    def __init__(self, cfg, img_name) :
        self.image_name = img_name
        self.cfg = cfg

        ##############################
        # 1. SETTINGS
        ##############################
        # set configs
        self._setConfigs(cfg)
        # set random seeds
        set_random_seed(is_gpu=not self.cpu)
        # set dirs
        self._setDirs()
        # copy the configs.py
        if os.path.isfile('./configs.py') :
            shutil.copy('./configs.py', cfg.save_dir)
        # set loggers
        self._setLoggers()
        # save and log configs
        for k, v in cfg.items() :
            log = '{} : {}'.format(k, v)
            self.config_logger.info(log)
        # set GPUs
        self._setGPUs()
        self.eps=1e-8
        # set images
        self.image_name = img_name
        self._setImages(img_name)
        
        ##############################
        # 2. NETWORK and INPUT NOISE
        ##############################
        # set models
        self.net = OrderedDict()
        # 1. DIP
        self.net['dip'] = get_net(**self.Generator).to(self.device)
        self.net['downsampler'] = Downsampler(**self.Downsampler).to(self.device)
        # 2. discriminator
        self.net['netD'] = WDiscriminator(**self.Discriminator).to(self.device)

        # set input noise
        self.net_input = get_noise(self.input_channels, self.input_type, self.img_HR_size).type(self.dtype).detach()
        self.noise = self.net_input.detach().clone()

        self.real_noise_input = get_noise(self.n_channels, self.input_type, self.img_LR_size, 'n', self.noise_level/255.).type(self.dtype).detach()

        ##############################
        # 3. OPTIMIZERS and LOSS
        ##############################
        # optimizers
        self._setOptimizers()
        # loss
        self.mse_loss = nn.MSELoss().type(self.dtype)

        self.start_epoch = 0
        self.curr_epoch = 0
        self.psnr_best = 0


    def train(self) :
        self.best_result = None
        self.save_image_out_HR = None
        self.save_image_out_LR = None
        self.pseudo_HR = None
        self.pseudo_LR = None
        self.prev_out_HR = None
        self.prev_out_LR = None

        for i in range(self.num_iter) :
            self.netDIP_loss = 0
            self.netD_loss = 0

            for k, v in self.optimizers.items() :
                v.zero_grad()

            self.optimizers['netD'].step(self.closure_disc)
            self.optimizers['dip'].step(self.closure_dip)

            self.get_currentResults(i)
            if i % self.show_every == 1 :
                self.save_images(i)
                self.plot_closure(i)

    #--------------------------------------------------------------

    def closure_dip(self) :

        if self.reg_std :
            self.net_input = self.net_input + (self.noise.normal_() * self.reg_std)
        
        self.requires_grad(self.net['netD'], True)
        self.requires_grad(self.net['dip'], True)

        self.image_out_HR = self.net['dip'](self.net_input)
        self.image_out_LR = self.net['downsampler'](self.image_out_HR)

        # calc reconstruction loss and backpropagate
        self.noise_out = self.image_out_LR - self.images['LR_noisy']

        self.recon_input = self.image_out_LR

        self.recon_loss = self.mse_loss(self.recon_input, self.images['LR_noisy'].type(self.dtype)) * self.optimizer_dip['recon_weight']

        self.recon_loss.backward(retain_graph=True)

        # calc adversarial loss and backpropagate
        self.noise_out = self.image_out_LR - self.images['LR_noisy']
        output = self.net['netD'](self.noise_out.detach())

        errG = -output.mean()
        self.adv_loss = errG * self.optimizer_dip['adv_weight']
        self.adv_loss.backward(retain_graph=True)

        # calc self-supervised loss and backpropagate
        if self.optimizer_dip['ssl_weight'] > 0 and self.pseudo_HR is not None:
            self.ssl_HR = self.mse_loss(self.image_out_HR, self.pseudo_HR)*self.optimizer_dip['ssl_weight']
            self.ssl_HR.backward(retain_graph=True)
        else : self.ssl_HR = 0

        if self.optimizer_dip['ssl_weight'] > 0 and self.pseudo_LR is not None:
            self.ssl_LR = self.mse_loss(self.image_out_LR, self.pseudo_LR)*self.optimizer_dip['ssl_weight']
            self.ssl_LR.backward(retain_graph=True)
        else : self.ssl_LR = 0

        # calc netDIP loss
        self.netDIP_loss = self.recon_loss + self.adv_loss + self.ssl_LR + self.ssl_HR


        # set self-supervised loss input
        if self.prev_out_HR is None :
            self.prev_out_HR = self.image_out_HR
            self.prev_out_LR = self.image_out_LR

        else :
            self.prev_out_HR = self.save_image_out_HR
            self.prev_out_LR = self.save_image_out_LR

        self.save_image_out_HR = self.image_out_HR
        self.save_image_out_LR = self.image_out_LR

        
        self.pseudo_HR = self.save_image_out_HR
        self.pseudo_LR = self.save_image_out_LR

        

        return self.netDIP_loss


    def closure_disc(self) :

        self.requires_grad(self.net['netD'], True)
        self.requires_grad(self.net['dip'], False)

        ###########################################
        # 1. train with real noise
        ###########################################
        
        output = self.net['netD'](self.real_noise_input)
        self.errD_real = -output.mean()

        self.errD_real.backward(retain_graph = True)
        
        ###########################################
        # 2. train with fake noise
        ###########################################

        # update dip input
        if self.reg_std > 0 :
            self.net_input = self.net_input + (self.noise.normal_() * self.reg_std)

        # get fake noise and loss
        self.image_out_HR = self.net['dip'](self.net_input)
        self.image_out_LR = self.net['downsampler'](self.image_out_HR)

        self.noise_out = self.image_out_LR - self.images['LR_noisy']

        output = self.net['netD'](self.noise_out.detach())
        self.errD_fake = output.mean()

        self.errD_fake.backward(retain_graph=True)

        # Add gradients from the all-real and all-fake
        self.netD_loss = -self.errD_real + self.errD_fake

        return self.netD_loss


    def save_images(self, step) :
        # get numpy RGB results
        final_image_HR = np.clip(self.current_result.learned_HR, 0, 1).transpose(1,2,0).copy()*255

        # save clean HR image
        filename = os.path.join(self.result_dir, self.image_name+"_{:04d}_cleaned.png".format(step))        
        io.imsave(filename, final_image_HR.astype(np.uint8))


    def get_currentResults(self, step) :
        image_out_HR_np = np.clip(torch_to_np(self.image_out_HR), 0, 1)
        image_out_LR_np = np.clip(torch_to_np(self.image_out_LR), 0, 1)
        noise_out_np = np.clip(torch_to_np(self.noise_out), 0, 1)
        psnr_HR = peak_signal_noise_ratio(torch_to_np(self.images['HR']), image_out_HR_np)
        self.current_result = DenoizeResult(learned_HR=image_out_HR_np, learned_LR=image_out_LR_np, n=noise_out_np, psnr=psnr_HR, step=step)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result


    def plot_closure(self, step) :
        print('step : %d netDIP[rec : %f adv : %f sslHR : %f sslLR : %f ] netD[fake : %f real : %f ] psnr[curr : %f best : %f ]' % (step, 
                                                                                    self.recon_loss, self.adv_loss, self.ssl_HR, self.ssl_LR,
                                                                                    self.errD_fake, self.errD_real, 
                                                                                    self.current_result.psnr, self.best_result.psnr))

        print('step : %d netDIP[rec : %f adv : %f sslHR : %f sslLR : %f ] netD[fake : %f real : %f ] psnr[curr : %f best : %f ]' % (step, 
                                                                                    self.recon_loss, self.adv_loss, self.ssl_HR, self.ssl_LR,
                                                                                    self.errD_fake, self.errD_real, 
                                                                                    self.current_result.psnr, self.best_result.psnr), file=self.file)             


    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag


    def finalize(self) :
        final_image_HR = np.clip(self.best_result.learned_HR, 0, 1).transpose(1,2,0).copy()*255
        filename = os.path.join(self.valid_dir, self.image_name+"_cleaned_best.png")        
        io.imsave(filename, final_image_HR.astype(np.uint8))

        final_image_HR = np.clip(self.current_result.learned_HR, 0, 1).transpose(1,2,0).copy()*255
        filename = os.path.join(self.valid_dir, self.image_name+"_cleaned_final.png")        
        io.imsave(filename, final_image_HR.astype(np.uint8))

        print('best result info : [max psnr] : %f    [iteration] : %d' % (self.best_result.psnr, self.best_result.step) , file=self.file )
        print('best result info : [max psnr] : %f    [iteration] : %d' % (self.best_result.psnr, self.best_result.step) )
        
        self.file.close()

    #-----------------------------------------------------------------------------------------------------------------------------------------------
   
    def _setConfigs(self, cfg) :
        for key in cfg :
            setattr(self, key, cfg[key])


    def _setDirs(self) :
        self.valid_dir = os.path.join(self.save_dir, 'valid', self.dataset)
        self.result_dir = os.path.join(self.save_dir, self.dataset)
        make_dir(self.save_dir)
        make_dir(self.result_dir)
        make_dir(self.valid_dir)
        #tensorboard
        self.log_dir = os.path.join(self.save_dir, 'tensorboard')
        make_dir(self.log_dir)

        # self.board_writer = SummaryWriter(self.log_dir)
        self.file = open(os.path.join(self.result_dir, self.image_name+'_psnr.txt'), 'w')


    def _setLoggers(self) :
        setup_logger('configs', self.save_dir, 'configs', level=logging.INFO, screen=True)
        setup_logger('valid', self.save_dir, 'valid', level=logging.INFO, is_formatter=True, screen=False)
        self.config_logger = logging.getLogger('configs')    #training logger
        self.valid_logger = logging.getLogger('valid')      #validation logger


    def _setGPUs(self) :
        if self.cpu :
            self.device = torch.device('cpu')
        else :
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            self.device = torch.device('cuda')


    def _setImages(self, image_name) :
        print(image_name)
        image_path = os.path.join(self.data_dir, self.dataset)
        img_orig_pil, img_orig_np = get_image(image_path+"/clean/"+image_name+".png", -1)

        # get rescaled clean HR image
        new_size = (img_orig_pil.size[0] - img_orig_pil.size[0] % 8, 
                    img_orig_pil.size[1] - img_orig_pil.size[1] % 8)

        bbox = [
                (img_orig_pil.size[0] - new_size[0])/2, 
                (img_orig_pil.size[1] - new_size[1])/2,
                (img_orig_pil.size[0] + new_size[0])/2,
                (img_orig_pil.size[1] + new_size[1])/2,
        ]

        img_HR_pil = img_orig_pil.crop(bbox)
        img_HR_np = pil_to_np(img_HR_pil)

        # get LR images
        img_LR_clean_pil, img_LR_clean_np = get_image(image_path+"/x"+str(self.factor)+"_"+str(self.noise_level)+"/"+image_name+"_clean.png")
        img_LR_noisy_pil, img_LR_noisy_np = get_image(image_path+"/x"+str(self.factor)+"_"+str(self.noise_level)+"/"+image_name+"_noisy.png")
        _, img_gt_np = get_image(image_path+"/x"+str(self.factor)+"_"+str(self.noise_level)+"/"+image_name+"_gt.png")

        self.img_HR_size = (img_HR_pil.size[1], img_HR_pil.size[0])
        self.img_LR_size = (img_LR_noisy_pil.size[1], img_LR_noisy_pil.size[0])

        print('HR and LR resolutions: %s, %s' % (str(img_HR_pil.size), str(img_LR_noisy_pil.size)))

        self.images = {
                'orig':  np_to_torch(img_orig_np).type(self.dtype),
                'LR_clean': np_to_torch(img_LR_clean_np).type(self.dtype),
                'LR_noisy': np_to_torch(img_LR_noisy_np).type(self.dtype),
                'noise_gt' : np_to_torch(img_gt_np).type(self.dtype),
                'HR': np_to_torch(img_HR_np).type(self.dtype),
           }

        self.n_channels = self.images['HR'].shape[1]
        self.Discriminator['nc'] = self.n_channels
        self.Generator['n_channels'] = self.n_channels
        self.Downsampler['n_planes'] = self.n_channels
        print('n_channels : ', self.n_channels)


    def  _setOptimizers(self) :
        optimizers = OrderedDict()

        for net_type, _ in self.net.items() :
            # 1. optimizer of discriminator
            if net_type == 'netD' :
                if self.optimizer_disc['type'] == 'adam' :
                    optimizers[net_type] = torch.optim.Adam(
                        self.net['netD'].parameters(), lr=self.optimizer_disc['LR']
                    )
                else :
                    assert False

            # 2. optimizer of DIP
            if net_type == 'dip' :
                if self.optimizer_dip['type'] == 'adam' :
                    optimizers[net_type] = torch.optim.Adam(
                            self.net['dip'].parameters(), lr = self.optimizer_dip['LR']
                    )

                else :
                    assert False
        
        self.optimizers = optimizers
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    