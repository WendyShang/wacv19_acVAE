# Attentive Attribute Conditioned Channel-Recurrent Autoencoding 
## Prerequisites
  - Linux, NVIDIA GPU + CUDA CuDNN 
  - Install torch dependencies from https://github.com/torch/distro
  - Install torch pacakge `cudnn`
```bash
luarocks install cudnn
```
  - Install the **batchDisc** branch of the git repo [stnbhwd](https://github.com/qassemoquab/stnbhwd/tree/batchDisc), as we need the batch discrimination layer. 

## Dataset
  - We provide code to train on 64x64 and 128x128 CelebA dataset. The processed images can be downloaded from [here](https://www.dropbox.com/s/dq17fdvc0j9hji0/celeba_align_loose_crop.tar.gz?dl=0).
  - For dataloading purpose, a t7 file needs to be placed under `gen/` folder and can be downloaded from [here](https://www.dropbox.com/s/grdyx11gif0v5uv/celeba.t7?dl=0).

## 1st Stage Training (64x64)
  - To train with conditional GAN,
```bash
th main.lua -data [data_path] -save [save_path] -dataset celeba -LR 0.0002 -latentType cgan -eps 1e-6 -mom 0.9 -step 60 -manualSeed 96 -attrDim 40 -beta1 0.5
```
  - To train with Conditional VAE-GAN (CVAE-GAN)
```bash
th main.lua -data [data_path] -save [save_path] -dataset celeba -nGPU 1 -alpha 0.0003 -LR 0.0003 -latentType cvaegan_pggan_pretrain -stage 1 -batchSize 128 -eps 1e-6 -mom 0.9 -step 60 -nEpochs 150 -manualSeed 196 -print_freq 100 -beta1 0.5 -beta 0.0025 -fakeLabel 4
```
  - To train with Channel-Recurrent Conditional VAE-GAN (CRVAE-GAN)
```bash
th main.lua -data [data_path] -save [save_path] -dataset celeba -nGPU 1 -alpha1 0.0003 -alpha2 0.0002 -LR 0.001 -latentType crvaegan_pggan_pretrain -stage 1 -batchSize 128 -eps 1e-6 -mom 0.9 -step 60 -nEpochs 150 -manualSeed 196 -print_freq 100 -beta1 0.5 -beta 0.0025 -kappa 0.01 -fakeLabel 4 -timeStep 8
```
  - To train with Attentive Channel-Recurrent Conditional VAE-GAN (ACVAE-GAN)
```bash
th main.lua -data [data_path] -save [save_path] -dataset celeba -nGPU 1 -alpha1 0.0003 -alpha2 0.0002 -LR 0.001 -latentType acvaegan_pggan_pretrain -stage 1 -batchSize 128 -eps 1e-6 -mom 0.9 -step 60 -nEpochs 150 -manualSeed 196 -print_freq 100 -beta1 0.5 -beta 0.0025 -kappa 0.01 -fakeLabel 4 -timeStep 8 -rho 0.05 -rho_entreg 0.05
```

## 2nd Stage Training (128x128)
  - For the 2nd stage training, models are initialized from the respective models from the 1st stage training.
  - To train with Progressive-Growing Conditional VAE-GAN (CVAE-GAN)
```bash
th main.lua -data [data_path] -save [save_path] -dataset celeba -nGPU 1 -alpha 0.0003 -LR 0.0001 -latentType cvaegan_pggan -stage 2 -batchSize 64 -eps 1e-6 -mom 0.9 -step 60 -nEpochs 150 -manualSeed 196 -print_freq 200 -beta1 0.9 -beta 0.0025 -fakeLabel 4 -init_weight_from [pretrained_model]
```
  - To train with Progressive-Growing Channel-Recurrent Conditional VAE-GAN (CRVAE-GAN)
```bash
th main.lua -data [data_path] -save [save_path] -dataset celeba -nGPU 1 -alpha1 0.0003 -alpha2 0.0002 -LR 0.0003 -latentType crvaegan_pggan -stage 2 -batchSize 64 -eps 1e-6 -mom 0.9 -step 60 -nEpochs 150 -manualSeed 196 -print_freq 200 -beta1 0.9 -beta 0.0025 -fakeLabel 4 -timeStep 8 -kappa 0.01 -init_weight_from [pretrained_model]
```
  - To train with Progressive-Growing Attentive Channel-Recurrent Conditional VAE-GAN (ACVAE-GAN)
```bash
th main.lua -data [data_path] -save [save_path] -dataset celeba -nGPU 1 -alpha1 0.0003 -alpha2 0.0002 -LR 0.0003 -latentType acvaegan_pggan -stage 2 -batchSize 64 -eps 1e-6 -mom 0.9 -step 60 -nEpochs 150 -manualSeed 196 -print_freq 200 -beta1 0.9 -beta 0.0025 -fakeLabel 4 -timeStep 8  -kappa 0.01 -rho 0.05 -rho_entreg 0.05 -init_weight_from [pretrained_model]
```

## Citation
If you find our code useful, please cite our paper [[pdf](http://www-personal.umich.edu/~shangw/wacv19.pdf)]:
```
@inproceedings{shang2017attentive,
  title={Attentive Conditional Channel-Recurrent Autoencoding for Attribute-Conditioned Face Synthesis},
  author={Shang, Wenling and Sohn, Kihyuk},
  booktitle={WACV},
  year={2019}
}
```
If you use the celebA dataset, please also cite the following paper
```
@inproceedings{liu2015faceattributes,
 author = {Ziwei Liu and Ping Luo and Xiaogang Wang and Xiaoou Tang},
 title = {Deep Learning Face Attributes in the Wild},
 booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
 month = December,
 year = {2015} 
}
```


## Acknowledgments
Torch is a **fantastic framework** for deep learning research, which allows fast prototyping and easy manipulation of gradient propogations. We would like to thank the amazing Torch developers and the community. Our implementation has especially been benefited from the following excellent repositories:
 - Variational Autoencoders: https://github.com/y0ast/VAE-Torch
 - Spatial Transformer Network: https://github.com/qassemoquab/stnbhwd
 - StackGAN: https://github.com/hanzhanggit/StackGAN
 - facebook.resnet.torch: https://github.com/facebook/fb.resnet.torch
 - DCGAN: https://github.com/soumith/dcgan.torch
 - Generating Faces with Torch: https://github.com/skaae/torch-gan
 - Attr2Img: https://github.com/xcyan/eccv16_attr2img
 - CIFAR10: https://github.com/szagoruyko/cifar.torch  
