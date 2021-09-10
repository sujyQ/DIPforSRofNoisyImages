from .common_utils import *
from .denoising_utils import *

def put_in_center(img_np, target_size):
    img_out = np.zeros([3, target_size[0], target_size[1]])
    
    bbox = [
            int((target_size[0] - img_np.shape[1]) / 2),
            int((target_size[1] - img_np.shape[2]) / 2),
            int((target_size[0] + img_np.shape[1]) / 2),
            int((target_size[1] + img_np.shape[2]) / 2),
    ]
    
    img_out[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] = img_np
    
    return img_out


# def load_LR_HR_imgs_sr(fname, imsize, factor, enforse_div32=None, denoising=True, sigma=25):
#     '''Loads an image, resizes it, center crops and downscales.

#     Args: 
#         fname: path to the image
#         imsize: new size for the image, -1 for no resizing
#         factor: downscaling factor
#         enforse_div32: if 'CROP' center crops an image, so that its dimensions are divisible by 32.
#         denoising : if True, make HR and LR noisy
#         sigma : noising factor
#     '''
#     img_orig_pil, img_orig_np = get_image(fname, -1)
    
#     n_channels = int(img_orig_np.size/img_orig_pil.size[0]/img_orig_pil.size[1])
#     print('channels : ', n_channels)
    
#     if imsize != -1:
#         img_orig_pil, img_orig_np = get_image(fname, imsize)
        
#     # For comparison with GT
#     if enforse_div32 == 'CROP':
#         new_size = (img_orig_pil.size[0] - img_orig_pil.size[0] % 32, 
#                     img_orig_pil.size[1] - img_orig_pil.size[1] % 32)

#         bbox = [
#                 (img_orig_pil.size[0] - new_size[0])/2, 
#                 (img_orig_pil.size[1] - new_size[1])/2,
#                 (img_orig_pil.size[0] + new_size[0])/2,
#                 (img_orig_pil.size[1] + new_size[1])/2,
#         ]

#         img_HR_pil = img_orig_pil.crop(bbox)
#         img_HR_np = pil_to_np(img_HR_pil)
#     else:
#         img_HR_pil, img_HR_np = img_orig_pil, img_orig_np

#     LR_size = [
#                img_HR_pil.size[0] // factor, 
#                img_HR_pil.size[1] // factor
#     ]

#     img_LR_pil = img_HR_pil.resize(LR_size, Image.ANTIALIAS)
#     img_LR_np = pil_to_np(img_LR_pil)

#     if denoising :
#         img_LR_pil, img_LR_np = get_noisy_image(img_LR_np, sigma/255.)

#     print('HR and LR resolutions: %s, %s' % (str(img_HR_pil.size), str (img_LR_pil.size)))

#     return {
#                 'orig_pil': img_orig_pil,
#                 'orig_np':  img_orig_np,
#                 'LR_pil':  img_LR_pil, 
#                 'LR_np': img_LR_np,
#                 'HR_pil':  img_HR_pil, 
#                 'HR_np': img_HR_np,
#                 'channels' : n_channels
#            }

def load_LR_HR_imgs_sr(path, image_name, imsize, factor):
    '''Loads an image, resizes it, center crops and downscales.

    Args: 
        fname: path to the image
        imsize: new size for the image, -1 for no resizing
        factor: downscaling factor
        enforse_div32: if 'CROP' center crops an image, so that its dimensions are divisible by 32.
    '''
    print(image_name)
    img_orig_pil, img_orig_np = get_image(path+"/clean/"+image_name+".png", imsize)
        
    # For comparison with GT
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

    img_LR_clean_pil, img_LR_clean_np = get_image(path+"/noisy_LR/"+image_name+"_clean.png", imsize)
    img_LR_noisy_pil, img_LR_noisy_np = get_image(path+"/noisy_LR/"+image_name+"_noisy.png", imsize)
    img_gt_pil, img_gt_np = get_image(path+"/noisy_LR/"+image_name+"_gt.png", imsize)

    # print('HR and LR resolutions: %s, %s' % (str(img_HR_pil.size), str (img_LR_clean_pil.size)))

    return {
                'orig_np':  img_orig_np,
                'LR_clean_np': img_LR_clean_np,
                'LR_noisy_np': img_LR_noisy_np,
                'noise_gt_np' : img_gt_np,
                'HR_np': img_HR_np,
           }


def get_baselines(img_LR_pil, img_HR_pil):
    '''Gets `bicubic`, sharpened bicubic and `nearest` baselines.'''
    img_bicubic_pil = img_LR_pil.resize(img_HR_pil.size, Image.BICUBIC)
    img_bicubic_np = pil_to_np(img_bicubic_pil)

    img_nearest_pil = img_LR_pil.resize(img_HR_pil.size, Image.NEAREST)
    img_nearest_np = pil_to_np(img_nearest_pil)

    img_bic_sharp_pil = img_bicubic_pil.filter(PIL.ImageFilter.UnsharpMask())
    img_bic_sharp_np = pil_to_np(img_bic_sharp_pil)

    return img_bicubic_np, img_bic_sharp_np, img_nearest_np



def tv_loss(x, beta = 0.5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    
    return torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))
