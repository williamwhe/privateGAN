# advGAN
### ....
## Prerequisites
- Linux or OSZX
- NVIDA GPU + CUDA CuDNN
- tensorflow 1.3
## Getting started
- train whitebox model
```
python train_models.py
```
Models are saved under 'models/' floder.  
- adversary attack
```
python generate_mnist.py --c 1  --ld 500 --H_lambda 10  --cgan_flag 1 --patch_flag 1 --G_lambda 10 --s_l 0 --t_l 1
```
Adversary images will be saved as 'result.png'

## parameters 
- ld adversary loss 

- H_lambda hinge loss 

- cgan_flag concat or not

- patch_flag patchGAN

- G_lambda generator loss

- s_l source label 

- t_l target label 