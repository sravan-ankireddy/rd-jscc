Official implementation of the paper "[Residual Diffusion Models for Variable-Rate Joint Source Channel Coding of MIMO CSI](https://arxiv.org/pdf/2505.21681)"

## Environment Setup

Create a conda environment with Python 3.10 and install the required packages:

```bash
conda create -n rdjscc python=3.10
pip install -r requirements.txt
```

## Training and Inference

### Example Training Commands

#### Single Bandwidth
```bash
python train.py --gpu 0 --n_denoising_steps 20 --data_name jscc_cost2100 --dim_mults 1 2 3 4 --embed_dim 64 --bandwidth 32 --mrl_weights 1 --lr 3e-4 --final_lr 1e-5 --log_checkpoint_step 25000 --training
```

#### Multi-Rate Optimization with Matryoshka Representation Learning (MRL)
```bash
python train.py --gpu 1 --n_denoising_steps 20 --data_name jscc_cost2100 --dim_mults 1 2 3 4 --embed_dim 64 --bandwidth 32 24 16 8 --mrl_weights 1 0.75 0.5 0.25 --lr 3e-4 --final_lr 1e-5 --log_checkpoint_step 25000 --training
```

### Inference

For inference, use the same commands but remove the `--training` flag. For example:

```bash
python train.py --gpu 0 --n_denoising_steps 20 --data_name jscc_cost2100 --dim_mults 1 2 3 4 --embed_dim 64 --bandwidth 32 --mrl_weights 1 --lr 3e-4 --final_lr 1e-5 --log_checkpoint_step 25000
```
