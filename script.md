# Training scripts

## 1st Stage Training (64x64)
  - To train with Conditional VAE (CVAE)
```bash
th main.lua -data [data_path] -save [save_path] -dataset celeba -nGPU 1 -alpha 0.0003 -LR 0.0003 -latentType cvae -stage 1 -batchSize 128 -eps 1e-6 -mom 0.9 -step 60 -nEpochs 150 -manualSeed 196 -print_freq 100
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
