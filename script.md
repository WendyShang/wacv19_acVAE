### Training scripts

- Conditional VAE (CVAE)

```bash
th main.lua -data [data_path] -save [save_path] -dataset celeba -alpha 0.0003 -LR 0.0003 -latentType cvae -eps 1e-6 -mom 0.9 -step 60 -manualSeed 196 -print_freq 100
```
