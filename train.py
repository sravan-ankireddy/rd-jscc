import os
import argparse
import logging
import random
import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
os.environ["WANDB_MODE"] = "disabled"
WANDB_API_KEY="855ca832b5cbd3e8780e9d84965edc4df07b8f68"

# @@@@@@@@@@@@@@@@@@add those to the pycharm configuration!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/home/sa53869/softwares/anaconda3/envs/nrx"
# os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/home/heasung/anaconda3/envs/tf2_sionna"

os.system('export LD_LIBRARY_PATH=/home/sa53869/softwares/anaconda3/envs/nrx/lib:$LD_LIBRARY_PATH')
os.system('export PATH=$PATH:/home/sa53869/softwares/anaconda3/envs/nrx/bin')

os.system('echo $PATH')

dataset_path = "/mnt/Data/datasets/csi"

import argparse

parser = argparse.ArgumentParser(description="values from bash script")
parser.add_argument("--gpu", type=int, default=0, help="cuda device id")
parser.add_argument("--n_embeddings", type=int, default=16, help="n_embeddings to control bit-rate")
parser.add_argument("--n_cl_floats", type=int, default=1, help="N_cl for continuous case")

parser.add_argument("--random_seed", type=int, default=42, help="random seed")

parser.add_argument("--z_channels", type=int, default=2)

parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
parser.add_argument("--final_lr", type=float, default=1e-5, help="final learning rate")
parser.add_argument("--decay", type=float, default=0.8)
parser.add_argument("--minf", type=float, default=0.2)
parser.add_argument("--optimizer", type=str, default="adam")

parser.add_argument('--training', action='store_true', help="Run the training phase if specified.")
parser.add_argument("--n_train_steps", type=int, default=int(1e6))
parser.add_argument("--scheduler_checkpoint_step", type=int, default=1000)
parser.add_argument("--log_checkpoint_step", type=int, default=1000)
parser.add_argument("--load_model", action='store_true', help="Load a pretrained model if specified.")
parser.add_argument("--load_step", action='store_true', help="Load at specific checkpoint step.")

parser.add_argument('--model', type=str, default='VQDiffusion', choices=["VQDiffusion", "CSINet", "CRNet"])

parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument('--pred_mode', type=str, default='x', help='prediction mode')
parser.add_argument('--loss_type', type=str, default='l2', help='type of loss')
parser.add_argument('--n_denoising_steps', type=int, default=4, help='number of iterations')
# parser.add_argument('--sample_steps', type=int, default=4, help='number of steps for sampling (for validation)')
parser.add_argument('--embed_dim', type=int, default=32, help='dimension of embedding')
parser.add_argument('--embd_type', type=str, default="01", help='timestep embedding type')
parser.add_argument('--dim_mults', type=int, nargs='+', default=[1, 2], help='dimension multipliers')
# parser.add_argument('--dim_mults', type=int, nargs='+', default=[2, 4, 8, 16, 32], help='dimension multipliers')

parser.add_argument('--context_dim_mults', type=int, nargs='+', default=[1, 2], help='context dimension multipliers')
parser.add_argument('--reverse_context_dim_mults', type=int, nargs='+', default=[2, 1],
                    help='reverse context dimension multipliers')
parser.add_argument('--context_channels', type=int, default=8, help='number of context channels')
# parser.add_argument('--bandwidth', type=int, default=16, help='number of uplink subcarriers used for CSI feedback')
parser.add_argument('--bandwidth', type=int, nargs='+', default=[256, 128, 64, 32, 16, 8], help='list of number of uplink subcarriers used for CSI feedback')
parser.add_argument('--mrl_weights', type=float, nargs='+', default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], help='list of MRL weights')
parser.add_argument('--use_weighted_loss', default=True, help='if use weighted loss')
parser.add_argument('--weight_clip', type=int, default=5, help='snr clip for weighted loss')
parser.add_argument('--use_mixed_precision', action='store_true', help='if use mixed precision')
parser.add_argument('--clip_noise', action='store_true', help='if clip the noise during sampling')
parser.add_argument('--val_num_of_batch', type=int, default=1, help='number of batches for validation')

parser.add_argument('--var_schedule', type=str, default='cosine', help='variance schedule')
parser.add_argument('--additional_note', type=str, default='', help='additional_note')
parser.add_argument('--aux_loss_type', type=str, default='l2', help='type of auxiliary loss')
parser.add_argument("--aux_weight", type=float, default=0, help="weight for aux loss")

parser.add_argument("--data_root", type=str, default=dataset_path, help="root of dataset")
parser.add_argument("--use_aux_loss_weight_schedule", action="store_true", help="if use aux loss weight schedule")

# parser.add_argument("--data_name", type=str, default="cost2100_outdoor", help="name of dataset", choices=["cost2100_outdoor", "cdl"])
# parser.add_argument("--use_side_info", type=bool, default=False)

parser.add_argument("--data_name", type=str, default="cost2100_outdoor", help="name of dataset", choices=["cost2100_outdoor", "cdl", "jscc", "jscc_cost2100", "jscc_complex"])
parser.add_argument("--use_side_info", type=bool, default=False)
parser.add_argument("--noisy_uplink", type=bool, default=True)

config = parser.parse_args()

gpu_num = int(config.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
random_seed = config.random_seed

import random
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
from tensorflow import keras
import numpy as np

tf.config.run_functions_eagerly(False)
# tf.compat.v1.disable_eager_execution()


# os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = str(random_seed)


bw_str = "_".join(map(str, config.bandwidth))
w_str = "_".join(map(str, config.mrl_weights))

save_path = (
    f"results_jsac_rebuttal_efficient_unet_eff/"
    f"enc_large_dec_med_unet_{len(config.dim_mults)}_steps_20_em_{config.embed_dim}/"
    f"bw_{bw_str}_w_{w_str}/"
    f"bs_{config.batch_size}_lr_{config.lr}_{config.final_lr}"
)

# Compute result and log directories
from global_config import ROOT_DIRECTORY
result_base = os.path.join(
    ROOT_DIRECTORY,
    save_path,
    config.data_name
)
log_dir = os.path.join(result_base, 'logs')
os.makedirs(log_dir, exist_ok=True)

# -----------------------------------
# Logger Configuration
# -----------------------------------
logger = logging.getLogger('CSI_Trainer')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console handler for INFO level and above
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler for DEBUG level and above
file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# -----------------------------------

# Echo PATH for debug
os.system('echo $PATH')

# Set GPU and seeds
os.environ["CUDA_VISIBLE_DEVICES"] = f"{config.gpu}"
random.seed(config.random_seed)
tf.random.set_seed(config.random_seed)
np.random.seed(config.random_seed)

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from data import load_data
from modules.denoising_diffusion import GaussianDiffusion

from modules.unet import Unet

from modules.trainer import Trainer
from modules.vq_compressor import VQCompressor

data_config = {
    "dataset_name": config.data_name,
    "batch_size": config.batch_size,
    "data_path": config.data_root,
    "sequence_length": 1,
    "img_size": 32,
    "img_channel": 2,
    "add_noise": False,
    "img_hz_flip": False,
}

model_name = (
    f"{data_config['dataset_name']}-{config.model}"
    f"{'-N_v{}'.format(config.n_embeddings) if config.model == 'VQDiffusion' else '-N_cl{}'.format(config.n_cl_floats)}"
    f"{'-sideinfo' if config.use_side_info else ''}"
    f"{config.additional_note}"
)

print('model name:')
print(model_name)


# def schedule_func(ep):
#     return max(config.decay ** ep, config.minf)
import math
def schedule_func(step):
    initial_lr = config.lr  # 1e-4 from argument
    eta_min = config.final_lr#1e-6
    T_max = int(config.n_train_steps/1000)  # Total steps for decay
    
    # Clamp step to T_max
    step = min(step, T_max)
    
    # Cosine decay calculation
    progress = step / T_max
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    
    # Calculate current LR
    current_lr = eta_min + (initial_lr - eta_min) * cosine_decay
    
    # Return as multiplier relative to initial LR
    return current_lr / initial_lr


def main():
    train_data, val_data, data_generator = load_data(data_config, batch_size=config.batch_size)

    input_img_shape = (2, 32, 32)

    if config.model == "VQDiffusion":
        context_model = VQCompressor(
            batch_size=config.batch_size,
            n_embeddings=config.n_embeddings,
            dim=config.embed_dim,
            dim_mults=config.context_dim_mults,
            reverse_dim_mults=config.reverse_context_dim_mults,
            channels=data_config["img_channel"],
            out_channels=config.context_channels,
            bandwidth=config.bandwidth, # list of bandwidths
        )
        # denoise_model = Unet(
        #     dim=config.embed_dim,
        #     channels=config.z_channels,
        #     context_channels=config.context_channels,
        #     dim_mults=config.dim_mults,
        #     context_dim_mults=reversed(config.reverse_context_dim_mults),
        #     embd_type=config.embd_type,
        # )
        ### FIX ME
        denoise_model = Unet(
            dim=config.embed_dim,
            channels=config.z_channels,
            context_channels=2,
            dim_mults=config.dim_mults,
            context_dim_mults=reversed([1, 1]),
            embd_type=config.embd_type,
        )

        model = GaussianDiffusion(
            denoise_fn=denoise_model,
            context_fn=context_model,
            input_img_shape=input_img_shape,
            batch_size=config.batch_size,
            n_denoising_steps=2,#config.n_denoising_steps,
            loss_type=config.loss_type,
            pred_mode=config.pred_mode,
            aux_loss_weight=config.aux_weight,
            aux_loss_type=config.aux_loss_type,
            var_schedule=config.var_schedule,
            use_loss_weight=config.use_weighted_loss,
            loss_weight_min=config.weight_clip,
            use_aux_loss_weight_schedule=config.use_aux_loss_weight_schedule,
            use_side_info=config.use_side_info,
            mrl_weights=config.mrl_weights,
        )
    elif config.model == "CSINet":
        from modules.models.csinet import CSINet
        model = CSINet(batch_size=config.batch_size, latent_dim=config.n_cl_floats, use_side_info=config.use_side_info)
    elif config.model == "CRNet":
        from modules.models.crnet import CRNet
        model = CRNet(batch_size=config.batch_size, latent_dim=config.n_cl_floats, use_side_info=config.use_side_info)


    # Trainer setup
    trainer = Trainer(
        rank=config.gpu,
        sample_steps=2,#config.n_denoising_steps,
        model=model,
        train_dl=train_data,
        val_dl=val_data,
        data_generator=data_generator,
        scheduler_function=schedule_func,
        scheduler_checkpoint_step=config.scheduler_checkpoint_step,
        train_lr=config.lr,
        train_num_steps=config.n_train_steps,
        save_and_sample_every=config.log_checkpoint_step,
        results_folder=os.path.join(result_base, "model", model_name),
        tensorboard_dir=os.path.join(result_base, "tensorboard", model_name),
        wandb_api_key=os.environ.get("WANDB_API_KEY"),
        wandb_project="csi_comp_mask",
        model_name=model_name,
        val_num_of_batch=config.val_num_of_batch,
        optimizer=config.optimizer,
        logger=logger
    )

    # Optional model loading
    if config.load_model:
        logger.info("Loading pretrained model...")
        trainer.load(idx=1, load_step=config.load_step)

    # Training
    if config.training:
        logger.info("Starting training...")
        trainer.train()

    # Inference
    logger.info(f"#### Running inference on 10,000 batches using trained model : {model_name} ####")
    trainer.predict()


if __name__ == "__main__":
    main()
