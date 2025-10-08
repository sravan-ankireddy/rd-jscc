import tensorflow as tf
import os
import gc
import numpy as np
import logging, io
from pathlib import Path
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.mixed_precision import global_policy as mixed_precision
import time
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

from tqdm import tqdm
import wandb

def batch_psnr(imgs1, imgs2):
    batch_mse = tf.reduce_mean(tf.square(imgs1 - imgs2), axis=(1, 2, 3))
    batch_psnr = 20 * tf.math.log(1.0 / tf.sqrt(batch_mse)) / tf.math.log(10.0)
    return tf.reduce_mean(batch_psnr)

def batch_nmse(compressed, batch):
    x_test_real = tf.reshape(batch[:, :, :, 0], [tf.shape(batch)[0], -1])
    x_test_imag = tf.reshape(batch[:, :, :, 1], [tf.shape(batch)[0], -1])
    x_test_C = tf.complex(x_test_real - 0.5, x_test_imag - 0.5)

    x_hat_real = tf.reshape(compressed[:, :, :, 0], [tf.shape(compressed)[0], -1])
    x_hat_imag = tf.reshape(compressed[:, :, :, 1], [tf.shape(compressed)[0], -1])
    x_hat_C = tf.complex(x_hat_real - 0.5, x_hat_imag - 0.5)

    # Compute power and MSE
    power = tf.reduce_sum(tf.abs(x_test_C) ** 2, axis=1)
    mse = tf.reduce_sum(tf.abs(x_test_C - x_hat_C) ** 2, axis=1)

    # Compute NMSE
    nmse = tf.reduce_mean(mse / power) #10 * tf.math.log(tf.reduce_mean(mse / power)) / tf.math.log(10.0)

    return nmse

def get_flops(model, input_shape):
    try:
        forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + input_shape[1:])])
        graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
        flops = graph_info.total_float_ops
        return flops
    except Exception as e:
        return None

class Trainer(object):
    def __init__(
        self,
        rank,
        sample_steps,
        model,
        train_dl,
        val_dl,
        data_generator,
        scheduler_function,
        train_lr=1e-4,
        train_num_steps=1000000,
        scheduler_checkpoint_step=100000,
        save_and_sample_every=1000,
        results_folder="./results",
        tensorboard_dir="./tensorboard_logs/diffusion-video/",
        wandb_api_key = None,
        wandb_project = None,
        model_name="model",
        val_num_of_batch=1,
        optimizer="adam",
        ema_decay=0.999,
        ema_update_interval=10,
        ema_step_start=100,
        use_mixed_precision=False,
        logger=None
    ):
        super().__init__()
        # Logger setup
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Initializing Trainer for model: {model_name}")

        self.model = model
        self.val_num_of_batch = val_num_of_batch
        self.sample_steps = sample_steps
        self.save_and_sample_every = save_and_sample_every

        self.train_num_steps = train_num_steps

        if train_dl is not None and val_dl is not None:
            self.train_dl = iter(train_dl.repeat())
            self.val_dl = iter(val_dl)
        else:
            self.train_dl = None
            self.val_dl = None
            self.data_generator = data_generator
            
        self.batch_size = 1
        if hasattr(self, 'data_generator'):
            input_shape = self.data_generator.get_batch().shape
            self.model.build(input_shape)
            self.batch_size = input_shape[0]
        else:
            input_shape = next(self.train_dl).shape
            self.model.build(input_shape)
            self.batch_size = input_shape[0]
            
        # Capture and log model summary
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + "\n"))
        summary_str = stream.getvalue()
        self.logger.info("Model Summary:\n%s", summary_str)
        
        # # Calculate and print FLOPs
        # flops = get_flops(self.model, input_shape)
        # if flops is not None:
        #     macs = flops / 2
        #     print(f"MACs: {macs:,}")
        #     print(f"FLOPs: {flops:,}")
        #     self.logger.info(f"MACs: {macs:,}")
        #     self.logger.info(f"FLOPs: {flops:,}")
        # else:
        #     print("Could not calculate FLOPs")
        #     self.logger.info("Could not calculate FLOPs")

        self.init_train_lr = train_lr
        if optimizer == "adam":
            self.opt = Adam(learning_rate=train_lr)
        elif optimizer == "adamw":
            self.opt = AdamW(learning_rate=train_lr)
        self.scheduler = scheduler_function
        self.ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        self.ema_update_interval = ema_update_interval
        self.ema_step_start = ema_step_start
        self.scaler = mixed_precision.LossScaleOptimizer(self.opt) if use_mixed_precision else None

        self.step = 0
        self.device = rank
        self.scheduler_checkpoint_step = scheduler_checkpoint_step

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self.writer = tf.summary.create_file_writer(tensorboard_dir)
        
        self.wandb_api_key = wandb_api_key
        self.wandb_project = wandb_project
        
        # Add W&B initialization
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
            wandb.init(
                project=self.wandb_project,
                config={
                    "learning_rate": train_lr,
                    "ema_decay": ema_decay,
                    "batch_size": self.batch_size
                }
            )



    def save(self):
        self.logger.debug(f"Saving model at step {self.step}")
        data = {
            "step": self.step,
            "model": self.model.get_weights(),
            "ema": {var.name: var.numpy() for var in self.model.trainable_variables}
        }
        idx = (self.step // self.save_and_sample_every) % 3
        np.save(os.path.join(self.results_folder, f"{self.model_name}_{idx}.npy"), data)
        
        # Save in TensorFlow checkpoint format
        checkpoint_path = os.path.join(self.results_folder, f"checkpoint_{self.model_name}_{idx}")
        self.model.save_weights(checkpoint_path)

    def load(self, idx=0, load_step=True):
        input_snr = tf.random.uniform(shape=[self.batch_size], minval=-10.0, maxval=10.0) # snr for uplink transmission
        compressed, target_img, target_img_sf, bps = self.predict_step(input_snr)

        data = np.load(os.path.join(self.results_folder, f"{self.model_name}_{idx}.npy"), allow_pickle=True).item()
        self.model.set_weights(data["model"])
        #ema_restore = {self.model.get_layer(name).trainable_variables[0]: value for name, value in data["ema"].items()}
        #self.ema.apply(ema_restore)
        if load_step:
            self.step = data["step"]

        self.logger.info(f"Load Success. (step:{self.step})")

    @tf.function(reduce_retracing=True)
    def update(self, data):
        print("tracing check def update(self, data)")
        with tf.GradientTape() as tape:
            # loss, aloss = self.model(data * 2.0 - 1.0, training=True)
            
            # do not normalize the input data, yet ( * 2.0 - 1.0)

            loss = self.model(data, training=True)
            
            #loss, aloss = self.model(data, training=True)

            total_loss = loss #+ 0.00045 * aloss
            if self.scaler:
                scaled_loss = self.scaler.get_scaled_loss(total_loss)
            else:
                scaled_loss = total_loss

        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)

        if self.scaler:
            scaled_gradients = self.scaler.get_unscaled_gradients(gradients)
            gradients = [tf.clip_by_norm(grad, 1.0) for grad in scaled_gradients]
            self.scaler.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.scaler.update()
        else:
            gradients = [tf.clip_by_norm(grad, 1.0) for grad in gradients]
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss #, aloss

    @tf.function
    def train_step(self):
        if self.train_dl is not None:
            data = next(self.train_dl)
        else:
            data = self.data_generator.get_batch()
        # loss, aloss = self.update(data)
        
        loss = self.update(data)

        return loss #, aloss

    @tf.function
    def predict_step(self, input_snr):
        if self.val_dl is not None:
            batch = next(self.val_dl)
        else:
            batch = self.data_generator.get_batch()

        # do not normalize the input data, yet ( * 2.0 - 1.0)
        compressed, bps = self.model.predict(batch, input_snr=input_snr, sample_steps=self.sample_steps)

        target_img = batch[...,1]
        target_img = target_img[:,:,:32]
        target_img_sf = batch[...,3]
        return compressed, target_img, target_img_sf, bps

    
    def predict(self):
        overall_ad_nmse = []
        overall_sf_nmse = []
        overall_bps = []
        B = self.batch_size

        snr_range = np.arange(-10.0, 11.0, 1.0)  # SNR values from -10.0 to 10.0 in steps of 1.0

        # Loop over SNR values from -10.0 to 10.0 in steps of 1.0
        for snr in snr_range:
            # Create an input_snr tensor for the current SNR value
            input_snr = tf.fill([B], tf.cast(snr, tf.float32))  # snr for uplink transmission

            ad_nmse_list = []
            sf_nmse_list = []
            bps_list = []
            for batch_idx in tqdm(range(1000)):
                start_time = time.time()
                compressed, target_img, target_img_sf, bps = self.predict_step(input_snr)
                # print("prediction takes {}sec".format(time.time() - start_time))
                compressed = (compressed + 1.0) * 0.5
                compressed = tf.transpose(compressed, perm=[0, 2, 3, 1])
                target_img = tf.transpose(target_img, perm=[0, 2, 3, 1])
                # matching the format in dataloader 
                target_img_sf = tf.transpose(target_img_sf, perm=[0, 3, 2, 1])
                
                compressed_clipped = tf.clip_by_value(compressed, 0.0, 1.0)
                batch_nmse_val_ad = batch_nmse(compressed_clipped, target_img)
                
                ad_nmse_list.append(batch_nmse_val_ad)
                bps_list.append(bps)
                
                ## SF nmse computations
                
                ## compute max_vals from unnormalized sf data
                
                # Convert the last dimension (real, imag) into a complex tensor.
                # Assumes target_img_sf has shape (N, 256, 32, 2).
                x_complex = tf.complex(target_img_sf[..., 0], target_img_sf[..., 1])  # Shape: (N, 256, 32)

                # Transpose to (N, 32, 256)
                x_complex = tf.transpose(x_complex, perm=[0, 2, 1])

                # Apply 2D IFFT
                x_ifft = tf.signal.ifft2d(x_complex)

                # Crop to center 32x32: taking the first 32 columns.
                x_cropped = x_ifft[:, :, :32]

                # Split into real and imaginary parts and rearrange to shape (N, 2, 32, 32)
                x_processed = tf.stack([tf.math.real(x_cropped), tf.math.imag(x_cropped)], axis=1)

                # Ensure the result is float32
                x_processed = tf.cast(x_processed, tf.float32)

                # Compute max values for normalization across axes (1, 2, 3)
                max_vals = tf.reduce_max(tf.abs(x_processed), axis=[1, 2, 3], keepdims=True)

                compressed_clipped_real = compressed_clipped[:, :, :, 0]
                compressed_clipped_imag = compressed_clipped[:, :, :, 1]
                
                compressed_processed = np.stack([compressed_clipped_real, compressed_clipped_imag], axis=1)
                
                # undo normalization
                max_vals = tf.convert_to_tensor(max_vals, dtype=tf.float32)
                compressed_processed -= 0.5
                compressed_processed *= (2 * max_vals)
                
                pad = tf.zeros([B, 2, 32, 256-32], dtype=tf.float32)
                compressed_processed = tf.concat([compressed_processed, pad], axis=3)
                compressed_processed = tf.complex(compressed_processed[:, 0, :, :], compressed_processed[:, 1, :, :])
                
                compressed_processed_fft = tf.signal.fft2d(compressed_processed)
                
                compressed_processed_sf = tf.stack([tf.math.real(compressed_processed_fft), tf.math.imag(compressed_processed_fft)], axis=3)
                
                # scale to [0, 1]
                compressed_processed_sf = compressed_processed_sf + 0.5
                
                target_img_sf = target_img_sf + 0.5
                # breakpoint()                
                compressed_processed_sf = tf.clip_by_value(compressed_processed_sf, 0.0, 1.0)
                
                batch_nmse_val_sf = batch_nmse(compressed_processed_sf, target_img_sf)
                
                sf_nmse_list.append(batch_nmse_val_sf)

            # Compute average metrics for the current SNR
            batch_ad_nmse_val = 10 * tf.math.log(tf.reduce_mean(ad_nmse_list)) / tf.math.log(10.0)
            batch_sf_nmse_val = 10 * tf.math.log(tf.reduce_mean(sf_nmse_list)) / tf.math.log(10.0)
            batch_bps_val = tf.reduce_mean(bps_list)
            self.logger.info("SNR {}: Validation AD NMSE: {}, Validation SF NMSE: {},  bps: {}".format(snr, batch_ad_nmse_val, batch_sf_nmse_val, batch_bps_val))
            # breakpoint()
            
            overall_ad_nmse.append(batch_ad_nmse_val)
            overall_sf_nmse.append(batch_sf_nmse_val)
            overall_bps.append(batch_bps_val)

        # print the overall results including SNR values
        snr_values = list(np.arange(-10.0, 11.0, 1.0))
        self.logger.info("SNR values: [{}]".format(", ".join(f"{snr:.1f}" for snr in snr_values)))
        self.logger.info("Overall AD NMSE: [{}]".format(", ".join(f"{v:.4f}" for v in overall_ad_nmse)))
        self.logger.info("Overall SF NMSE: [{}]".format(", ".join(f"{v:.4f}" for v in overall_sf_nmse)))

        # convert overall_bps to numpy array
        # print("Overall BPS: ", np.array(overall_bps))

        return overall_ad_nmse, overall_bps

    
    
    def train(self):
        
        input_snr_train = tf.random.uniform(shape=[self.batch_size], minval=-10.0, maxval=10.0) # snr for uplink transmission
        
        with tqdm(total=self.train_num_steps, desc="Training Progress", unit="step") as pbar:
            while self.step < self.train_num_steps:
                self.opt.learning_rate = self.init_train_lr * self.scheduler((self.step-self.scheduler_checkpoint_step)//self.scheduler_checkpoint_step)#self.save_and_sample_every) 
                
                loss = self.train_step()

                if self.step % int(self.save_and_sample_every) == 0:
                    nmse_list = []
                    for batch_idx in range(100):
                        # compressed, target_img, bps = self.predict_step(input_snr_train)
                        compressed, target_img, target_img_sf, bps = self.predict_step(input_snr_train)

                        compressed = (compressed + 1.0) * 0.5
                        compressed = tf.transpose(compressed, perm=[0, 2, 3, 1])
                        target_img = tf.transpose(target_img, perm=[0, 2, 3, 1])
                        batch_nmse_val = batch_nmse(tf.clip_by_value(compressed, 0.0, 1.0), target_img)
                        nmse_list.append(batch_nmse_val)
                    i = 0
                    # self.ema.apply(self.model.trainable_variables)

                    batch_nmse_val = 10 * tf.math.log(tf.reduce_mean(nmse_list)) / tf.math.log(10.0)

                    step_tmp = self.step // self.save_and_sample_every

                    self.logger.info(
                    f"Step {self.step} | NMSE {batch_nmse_val.numpy():.4f} dB | BPS {bps.numpy():.4f} | LR {self.opt.learning_rate.numpy():.6f}"
                )
                    with self.writer.as_default():
                        tf.summary.scalar(f"bps/num{i}", bps, step=self.step // self.save_and_sample_every)
                        tf.summary.scalar(f"psnr/num{i}", batch_psnr(tf.clip_by_value(compressed, 0.0, 1.0), target_img),
                                        step=self.step // self.save_and_sample_every)
                        tf.summary.scalar(f"nmse/num{i}", batch_nmse_val, step=self.step // self.save_and_sample_every)

                        # Recovered samples
                        x_hat_C = tf.complex(compressed[:, :, :, 0] - 0.5, compressed[:, :, :, 1] - 0.5)
                        x_test_C = tf.complex(target_img[:, :, :, 0] - 0.5, target_img[:, :, :, 1] - 0.5)

                        tf.summary.image(f"compressed/num{i}", tf.expand_dims(tf.abs(x_hat_C), axis=-1), step=self.step // self.save_and_sample_every)
                        tf.summary.image(f"original/num{i}", tf.expand_dims(tf.abs(x_test_C), axis=-1), step=self.step // self.save_and_sample_every)
                        
                        
                    # Add W&B logging
                    if self.wandb_api_key:
                        wandb.log({
                            "nmse": batch_nmse_val.numpy(),
                            "psnr": batch_psnr(tf.clip_by_value(compressed, 0.0, 1.0), target_img).numpy(),
                            "bpp": bps.numpy(),
                            "learning_rate": self.opt.learning_rate.numpy()
                        })
                        
                        # Log images
                        x_hat_abs = tf.abs(x_hat_C)  # Shape: [batch_size, H, W]
                        x_hat_rgb = tf.expand_dims(x_hat_abs, axis=-1)  # Add channel dim -> [batch_size, H, W, 1]
                        x_hat_rgb = tf.repeat(x_hat_rgb, 3, axis=-1)  # Convert to "RGB" -> [batch_size, H, W, 3]
                        
                        x_test_abs = tf.abs(x_test_C)  # Shape: [batch_size, H, W]
                        x_test_rgb = tf.expand_dims(x_test_abs, axis=-1)
                        x_test_rgb = tf.repeat(x_test_rgb, 3, axis=-1)

                        wandb.log({
                            "compressed": [wandb.Image(x_hat_rgb[0])],  # Log first sample in batch
                            "original": [wandb.Image(tf.abs(x_test_rgb[0]))]
                        })

                if self.step % self.save_and_sample_every == 0:

                    self.save()
                self.step += 1
                pbar.update(1)

        self.save()

        self.logger.info("Training completed.")