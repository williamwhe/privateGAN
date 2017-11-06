# privateGAN
A library to protect privacy of a target function `f` against an adversary function `g`.
## Prerequisites
- Linux
- NVIDA GPU with CUDA and CuDNN
- tensorflow 1.3
## Getting start 
- train whitbox model
```
python train_models.py
```
Models are saved under 'models/' floder.
- adversary attack
```
python generate_mnist.py --c 1  --ld 1000 --H_lambda 100 --G_lambda 250 --learning_rate 0.001 --cgan_flag 1 --patch_flag 1 --max_iteration 2000
```
Adversary images will be saved as 'result.png'

## parameters 
- `ld`: adversarial loss coefficient.
- `G_lambda`: Generative classifier loss coefficient.
- `H_lambda`: hinge loss coefficient.
- `cgan_flag`: concat fake data to real data or not.
- `patch_flag`: Use patchGAN or not.
- `s_l`: source label
- `t_l`: target label (attack target)