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
  - To train with conditional VAE, 
```bash
CUDA_VISIBLE_DEVICES=1 th main.lua -data /local/wshang/celeba_align_loose_crop/ -save /local/wshang/test_cond_crVAE/ -alpha 0.0001 -LR 0.0003 -eps 1e-6 -mom 0.9 -step 60 -latentType cvae -manualSeed 1196 -stage 1
/local/wshang/test_cond_crVAE/allconv_cvae_beta_0_LR_0.0003_alpha_0.0001_beta1_0.9_seed_1196/
``` 
  - To train with conditional GAN,
```bash
CUDA_VISIBLE_DEVICES=2 th main.lua -data /local/wshang/celeba_align_loose_crop/ -save /local/wshang/test_cond_crVAE/ -dataset celeba -LR 0.0002 -latentType cgan -eps 1e-6 -mom 0.9 -step 60 -manualSeed 96 -attrDim 40 -beta1 0.5
/local/wshang/test_cond_crVAE/allconv_cgan_beta_0_LR_0.0002_alpha_1_beta1_0.5_seed_96/
```
  - To train with conditional VAE-GAN, 
```bash
th main_mnist.lua -LR 0.0003 -alpha 0.001 -latentType baseline -dataset mnist_28x28 -baseChannels 32 -nEpochs 200 -eps 1e-5 -mom 0.1 -step 50 -save /path/to/save/ -dynamicMNIST /path/to/dynamics/mnist/ -binaryMNIST /path/to/binary/mnist/
```
  - To train with channel-recurrent conditional VAE-GAN, 
```bash
th main_mnist.lua -LR 0.0003 -alpha 0.001 -latentType conv -dataset mnist_28x28 -baseChannels 32 -nEpochs 200 -eps 1e-5 -mom 0.1 -step 50 -save /path/to/save/ -dynamicMNIST /path/to/dynamics/mnist/ -binaryMNIST /path/to/binary/mnist/
```
  - To train with attentive channel-recurrent conditional VAE-GAN, 
```bash
th main_mnist.lua -LR 0.003 -timeStep 8 -alpha 0.001 -latentType lstm -dataset mnist_28x28 -baseChannels 32 -nEpochs 200 -eps 1e-5 -mom 0.1 -step 50 -save /path/to/save/ -dynamicMNIST /path/to/dynamics/mnist/ -binaryMNIST /path/to/binayr/mnist/
```

## 2nd Stage Training (128x128)
We provide code to perform 2nd stage training on top of the 1st stage models for VAE-GAN, crVAE-GAN and acVAE-GAN using the progressive growing scheme on inference, generation and discriminator networks, provided that the training of the 1st stage models is completed. 
  - To train with conditional VAE-GAN, 
```bash
th main_mnist.lua -LR 0.0003 -alpha 0.001 -latentType baseline -dataset mnist_28x28 -baseChannels 32 -nEpochs 200 -eps 1e-5 -mom 0.1 -step 50 -save /path/to/save/ -dynamicMNIST /path/to/dynamics/mnist/ -binaryMNIST /path/to/binary/mnist/
```
  - To train with channel-recurrent conditional VAE-GAN, 
```bash
th main_mnist.lua -LR 0.0003 -alpha 0.001 -latentType conv -dataset mnist_28x28 -baseChannels 32 -nEpochs 200 -eps 1e-5 -mom 0.1 -step 50 -save /path/to/save/ -dynamicMNIST /path/to/dynamics/mnist/ -binaryMNIST /path/to/binary/mnist/
```
  - To train with attentive channel-recurrent conditional VAE-GAN, 
```bash
th main_mnist.lua -LR 0.003 -timeStep 8 -alpha 0.001 -latentType lstm -dataset mnist_28x28 -baseChannels 32 -nEpochs 200 -eps 1e-5 -mom 0.1 -step 50 -save /path/to/save/ -dynamicMNIST /path/to/dynamics/mnist/ -binaryMNIST /path/to/binayr/mnist/
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
