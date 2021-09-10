from easydict import EasyDict
import torch

def getConfigs():
    cfg = EasyDict() 
    #=======================
    # 1. Dataset & Save Path
    #=======================  
    cfg.save_dir = 'path/to/your/result/directory'
    cfg.keyword = 'test'
        
    cfg.data_dir = 'path/to/your/data/directory'
    cfg.dataset = 'set5'
    cfg.factor = 2
    cfg.noise_level = 25
    cfg.input_type = 'noise'
    cfg.reg_std = 0.03
    cfg.input_channels = 32

    cfg.dtype = torch.cuda.FloatTensor

    cfg.content_noise_var = 0.01

    cfg.recon_noise_var = 0.002

    #=======================
    # 2. CPU/GPU Setting
    #=======================  
    cfg.cpu = False

    #=======================
    # 3. Model setting 
    #=======================
    cfg.Generator = { 
        'input_depth' : 32,
        'NET_TYPE' : 'skip',
        'pad' : 'reflection',
        'skip_n33d' : [8, 16, 32, 64, 128],
        'skip_n33u' : [8, 16, 32, 64, 128],
        'skip_n11' : [0, 0, 0, 4, 4],
        # 'num_scales' : 5,
        'upsample_mode' : 'bilinear',
        'n_channels' : 3
    }

    cfg.Downsampler = { 
        'n_planes' : 3,
        'factor' :  cfg.factor,
        'kernel_type' : 'lanczos2',
        'phase' : 0.5,
        'preserve_size' : True
    }

    cfg.Discriminator = {
        'nc' : 3,
        'nfc' : 32,
        'min_nfc' : 32,
        'ker_size' : 3,
        'num_layer' : 5,
        'stride' : 1,
        'pad_size' : 0
    }

    #=======================
    # 3. Hyperparams 
    #=======================
    cfg.resume = False  

    #=========================
    # 4. optimizer
    #=========================
    cfg.optimizer_dip = {
        'type' : 'adam',
        'LR' : 1e-2,
        'recon_weight' : 1,
        'adv_weight' : 1.2,
        'ssl_weight' : 1
    }

    cfg.optimizer_disc = {
        'type' : 'adam',
        'LR' : 1e-4
    }

    cfg.num_iter = 2000
    cfg.show_every = 200

    return cfg