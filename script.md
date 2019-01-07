### Training scripts

- Conditional VAE (CVAE)

```bash
th main.lua -data [data_path] -save [save_path] -dataset celeba -alpha 0.0003 -LR 0.0003 -latentType cvae -eps 1e-6 -mom 0.9 -step 60 -manualSeed 196 -print_freq 100
```

- CVAE-GAN (PGGAN and pretrain)

```bash
pretrain: th main.lua -data [data_path] -save [save_path] -dataset celeba -nGPU 1 -alpha 0.0003 -LR 0.0003 -latentType cvaegan_pggan_pretrain -eps 1e-6 -mom 0.9 -step 60 -nEpochs 150 -manualSeed 196 -beta1 0.5 -beta 0.0025 -fakeLabel 4 -print_freq 100 -batchSize 128
```

- CRVAE-GAN (PGGAN and pretrain)

```bash
pretrain: th main.lua -data [data_path] -save [save_path] -dataset celeba -alpha1 0.0003 -alpha2 0.0002 -LR 0.001 -latentType crvaegan_pggan_pretrain -eps 1e-6 -mom 0.9 -step 60 -nEpochs 150 -manualSeed 196 -beta1 0.5 -beta 0.0025 -kappa 0.01 -fakeLabel 4 -print_freq 100 -timeStep 4
pretrain: th main.lua -data [data_path] -save [save_path] -dataset celeba -alpha1 0.0003 -alpha2 0.0002 -LR 0.001 -latentType crvaegan_pggan_pretrain -eps 1e-6 -mom 0.9 -step 60 -nEpochs 150 -manualSeed 196 -beta1 0.5 -beta 0.0025 -kappa 0.01 -fakeLabel 4 -print_freq 100 -timeStep 8
```

- ACVAE-GAN (PGGAN and pretrain)

```bash
pretrain: th main.lua -data [data_path] -save [save_path] -dataset celeba -alpha1 0.0003 -alpha2 0.0002 -LR 0.001 -latentType acvaegan_pggan_pretrain -eps 1e-6 -mom 0.9 -step 60 -nEpochs 150 -manualSeed 196 -beta1 0.5 -beta 0.0025 -kappa 0.01 -fakeLabel 4 -print_freq 100 -timeStep 4 -rho 0.05 -rho_entreg 0.05
pretrain: th main.lua -data [data_path] -save [save_path] -dataset celeba -alpha1 0.0003 -alpha2 0.0002 -LR 0.001 -latentType acvaegan_pggan_pretrain -eps 1e-6 -mom 0.9 -step 60 -nEpochs 150 -manualSeed 196 -beta1 0.5 -beta 0.0025 -kappa 0.01 -fakeLabel 4 -print_freq 100 -timeStep 8 -rho 0.05 -rho_entreg 0.05
```
