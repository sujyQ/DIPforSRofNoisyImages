# Deep Image Prior for Super Resolution of Noisy Images

This repository is the official implementation of [Deep Image Prior for Super Resolution of Noisy Images](https://doi.org/10.3390/electronics10162014). 

<img src="fig/fig2.png">

## Requirements

To install requirements:

- python >= 3.6
- pytorch >= 1.2.0
- numpy
- imageio

## Training

Run this command:

```train
python train.py
```

>📋  Check training settings via [configs.py](https://github.com/sujyQ/DIPforSRofNoisyImages/blob/42b1d4d6d9ab4cfd651cc9cf2620ddb0dc5ada7b/configs.py)

## Quantitative Results

### PSNR | SSIM
<img src="fig/tab1.png">

## Qualitative Results

### X2, sig=15
<img src="fig/fig3.png">

### X2, sig=25
<img src="fig/fig4.png">

### X4, sig=15
<img src="fig/fig5.png">

### X4, sig=25
<img src="fig/fig6.png">

## Contributing

>📋  This code is built on [DIP](https://github.com/DmitryUlyanov/deep-image-prior). We thank the authors for sharing the code.

## Citation

>@article{han2021deep,
>  title={Deep Image Prior for Super Resolution of Noisy Image},
>  author={Han, Sujy and Lee, Tae Bok and Heo, Yong Seok},
>  journal={Electronics},
>  volume={10},
>  number={16},
>  pages={2014},
>  year={2021},
>  publisher={Multidisciplinary Digital Publishing Institute}
>}