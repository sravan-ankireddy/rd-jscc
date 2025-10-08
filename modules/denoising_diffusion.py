import time

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, losses
from functools import partial
from tqdm import tqdm
from .utils import cosine_beta_schedule, linear_beta_schedule, extract

from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

class GaussianDiffusion(tf.keras.Model):
    def __init__(self, denoise_fn, context_fn, input_img_shape, batch_size, use_side_info,  n_denoising_steps=4,
                 loss_type="l1", pred_mode="noise",
                 var_schedule="linear", aux_loss_weight=0, aux_loss_type="l1",
                 use_loss_weight=True, loss_weight_min=5,
                 use_aux_loss_weight_schedule=False,
                 mrl_weights=None):
        super(GaussianDiffusion, self).__init__()
        self.denoise_fn = denoise_fn
        self.context_fn = context_fn
        self.input_img_shape = input_img_shape
        self.batch_size = batch_size
        self.use_side_info = use_side_info
        self.loss_type = loss_type
        #self.lagrangian_beta = lagrangian
        self.var_schedule = var_schedule
        self.sample_steps = None
        self.aux_loss_weight = aux_loss_weight
        self.aux_loss_type = aux_loss_type
        self.use_aux_loss_weight_schedule = use_aux_loss_weight_schedule
        self.mrl_weights = mrl_weights
        assert pred_mode in ["noise", "x"]
        self.pred_mode = pred_mode
        self.use_loss_weight = use_loss_weight
        self.loss_weight_min = float(loss_weight_min)

        if var_schedule == "cosine":
            train_betas = cosine_beta_schedule(n_denoising_steps)
        elif var_schedule == "linear":
            train_betas = linear_beta_schedule(n_denoising_steps)

        train_alphas = 1.0 - train_betas
        train_alphas_cumprod = np.cumprod(train_alphas, axis=0)

        self.n_denoising_steps = tf.convert_to_tensor(train_betas.shape[0], dtype=tf.int32)#int(train_betas.shape[0])
        self.n_denoising_steps_float = tf.convert_to_tensor(train_betas.shape[0], dtype=tf.float32)#int(train_betas.shape[0])

        self.train_snr = tf.convert_to_tensor(train_alphas_cumprod / (1 - train_alphas_cumprod), dtype=tf.float32)
        self.train_snr = tf.clip_by_value(self.train_snr, 0,  self.loss_weight_min) if self.loss_weight_min > 0 \
            else tf.clip_by_value(self.train_snr, self.loss_weight_min, 0)

        self.train_betas = tf.convert_to_tensor(train_betas, dtype=tf.float32)
        self.train_alphas_cumprod = tf.convert_to_tensor(train_alphas_cumprod, dtype=tf.float32)
        self.train_sqrt_alphas_cumprod = tf.convert_to_tensor(np.sqrt(train_alphas_cumprod), dtype=tf.float32)
        self.train_sqrt_one_minus_alphas_cumprod = tf.convert_to_tensor(np.sqrt(1.0 - train_alphas_cumprod), dtype=tf.float32)
        self.train_sqrt_recip_alphas_cumprod = tf.convert_to_tensor(np.sqrt(1.0 / train_alphas_cumprod), dtype=tf.float32)
        self.train_sqrt_recipm1_alphas_cumprod = tf.convert_to_tensor(np.sqrt(1.0 / train_alphas_cumprod - 1), dtype=tf.float32)

        self.training = True

        self.x_start_shape = (self.batch_size, 2, 32, 32)
        self.noise_shape = (self.batch_size, 2, 32, 32)
        self.img_batch_shape = (self.batch_size, 2, 32, 32)

        #self.build(input_shape=self.img_batch_shape)
        
    def parameters(self, skip_keywords=["loss_fn_vgg", "ae_fn"], recurse=True):
        for var in self.trainable_variables:
            use = True
            for keyword in skip_keywords:
                if keyword in var.name:
                    use = False
                    break
            if use:
                yield var

    # @tf.function(reduce_retracing=True)
    # def get_extra_loss(self):
    #     print("tracing check get_extra_loss")
    #     return self.context_fn.get_extra_loss()

    #@tf.function(reduce_retracing=True)
    def set_sample_schedule(self, sample_steps):
        self.sample_steps = sample_steps
        if sample_steps != 1:
            indice = tf.cast(tf.linspace(0, self.n_denoising_steps - 1, sample_steps), tf.int32)
        else:
            indice = tf.convert_to_tensor([self.n_denoising_steps - 1], dtype=tf.int32)
        self.alphas_cumprod = tf.gather(self.train_alphas_cumprod, indice)
        self.snr = tf.gather(self.train_snr, indice)
        self.index = tf.gather(tf.range(self.n_denoising_steps, dtype=tf.int32), indice)
        self.alphas_cumprod_prev = tf.pad(self.alphas_cumprod[:-1], [[1, 0]], constant_values=1.0)
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = tf.sqrt(self.alphas_cumprod_prev)
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.one_minus_alphas_cumprod_prev = 1.0 - self.alphas_cumprod_prev
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod_prev = tf.sqrt(1.0 - self.alphas_cumprod_prev)
        self.sqrt_recip_alphas_cumprod = tf.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod_prev = tf.sqrt(1.0 / self.alphas_cumprod_prev)
        self.sqrt_recipm1_alphas_cumprod = tf.sqrt(1.0 / self.alphas_cumprod - 1)
        self.sigma = self.sqrt_one_minus_alphas_cumprod_prev / self.sqrt_one_minus_alphas_cumprod * tf.sqrt(1.0 - self.alphas_cumprod / self.alphas_cumprod_prev)


    def predict_noise_from_start(self, x_t, t, x0):
        # return (tf.gather(self.sqrt_recip_alphas_cumprod, t) * x_t - x0) / tf.gather(self.sqrt_recipm1_alphas_cumprod, t)
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t,batch_size=self.batch_size, x_shape= self.x_start_shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t,batch_size=self.batch_size, x_shape= self.x_start_shape)
        )

    def predict_v(self, x_start, t, noise):
        if self.training:
            return (extract(self.train_sqrt_alphas_cumprod, t, self.x_start_shape) * noise -
                    extract(self.train_sqrt_one_minus_alphas_cumprod, t, self.x_start_shape) * x_start)
        else:
            return (extract(self.sqrt_alphas_cumprod, t, self.x_start_shape) * noise -
                    extract(self.sqrt_one_minus_alphas_cumprod, t, self.x_start_shape) * x_start)

    def predict_start_from_v(self, x_t, t, v):
        if self.training:
            return (extract(self.train_sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.train_sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v)
        else:
            return (extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v)

    def predict_start_from_noise(self, x_t, t, noise):
        if self.training:
            return (extract(self.train_sqrt_recip_alphas_cumprod, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * x_t -
                    extract(self.train_sqrt_recipm1_alphas_cumprod, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * noise)
        else:
            return (extract(self.sqrt_recip_alphas_cumprod, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * noise)

    @tf.function(reduce_retracing=True)
    def ddim(self, x, t, context_output, clip_denoised, eta=0):
        print("tracing check def ddim(self, x, t,")
        if self.denoise_fn.embd_type == "01":
            #asdf = tf.gather(self.index,t)
            tt = tf.expand_dims(tf.cast(tf.gather(self.index, t), tf.float32), -1) / self.n_denoising_steps_float
            fx = self.denoise_fn(x, tt, context_output)

        else:
            fx = self.denoise_fn(x, tf.gather(self.index,t), context_output)
        if self.pred_mode == "noise":
            x_recon = self.predict_start_from_noise(x, t, fx)
        elif self.pred_mode == "x":
            x_recon = fx

        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, -1.0, 1.0)
        noise = fx if self.pred_mode == "noise" else self.predict_noise_from_start(x, t, x_recon)
        x_next = (
                extract(self.sqrt_alphas_cumprod_prev, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * x_recon +
                  tf.sqrt(
                      tf.clip_by_value(extract(self.one_minus_alphas_cumprod_prev, t, batch_size=self.batch_size, x_shape=self.x_start_shape) -
                           (eta * extract(self.sigma, t, batch_size=self.batch_size, x_shape=self.x_start_shape)) ** 2,clip_value_min=0, clip_value_max=1.0)) * noise +
                  eta * extract(self.sigma, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * tf.random.normal(self.noise_shape))
        return x_next

    @tf.function(reduce_retracing=True)
    def p_sample(self, x, t, context_output, clip_denoised, eta=0):
        print("tracing check p_sample")
        return self.ddim(x, t, context_output, clip_denoised, eta)

    @tf.function(reduce_retracing=True)
    def p_sample_loop(self, shape, context_output, clip_denoised=False, init=None, eta=0):
        print("tracing check p_sample_loop")
        b = shape[0]
        img = tf.zeros(shape) if init is None else init
        for i in tqdm(reversed(range(self.sample_steps)), desc="sampling loop time step", total=self.sample_steps, disable=True):
            time = tf.fill([b], i)
            img = self.p_sample(img, time, context_output, clip_denoised, eta)
        return img

    #@tf.function(reduce_retracing=True)
    def predict(self, images, input_snr = None, sample_steps=None, bpp_return_mean=True, init=None, eta=0):
        self.training = False
        compress_start_time = time.time()


        input_img, target_img, side_info, input_img_freq = tf.split(images, num_or_size_splits=4, axis=-1)
        input_img = input_img[:,:,:32]
        target_img = target_img[:,:,:32]
        side_info = side_info[:,:,:32]
        
        input_img = tf.squeeze(input_img, axis=-1)
        target_img = tf.squeeze(target_img, axis=-1)

        # center the data 
        input_img = input_img * 2.0 - 1.0
        target_img = target_img * 2.0 - 1.0

        # context_dict = self.context_fn(input_img)
        # context_dict_output, context_dict_bps, context_dict_q_latent, context_dict_a_loss = self.context_fn(input_img)
        
        # leave the uplink channel info as it is
        input_h = side_info # nearby uplink channel info used for simulating the uplink transmission
        if input_snr is None:
            B = tf.shape(input_img)[0]
            input_snr = tf.random.uniform(shape=[B], minval=-10.0, maxval=10.0) # snr for uplink transmission
        
        context, context_dict_output, context_dict_bps, context_dict_q_latent = self.context_fn(input_img, input_h, input_snr)

        # performance at largest bandwidth
        ind = 0
        context_dict_output = context_dict_output[ind]

        if self.use_side_info is True:
            side_info = tf.squeeze(side_info, axis=-1)
            context_dict_output[0] = tf.concat([context_dict_output[0], side_info], axis=1)

        #print("context_fn {}sec".format(time.time() - compress_start_time))
        compress_start_time = time.time()
        self.set_sample_schedule(self.n_denoising_steps if sample_steps is None else sample_steps)
        #print("set_sample_schedule takes {}sec".format(time.time() - compress_start_time))

        ## fix me
        # context_dict_output = [context_dict_output]

        compress_start_time = time.time()
        compressed_image = self.p_sample_loop(self.img_batch_shape, context_dict_output, clip_denoised=True, init=init,
                                              eta=eta)
        bpp = tf.reduce_mean(context_dict_bps) if bpp_return_mean else context_dict_bps
        self.training = True
        #print("encode/decode takes {} sec".format(time.time() - compress_start_time))
        return compressed_image, bpp

    @tf.function(reduce_retracing=True)
    def q_sample(self, x_start, t, noise):
        print("tracing check q_sample")
        sample = (extract(self.train_sqrt_alphas_cumprod, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * x_start
                  + extract(self.train_sqrt_one_minus_alphas_cumprod, t, batch_size=self.batch_size, x_shape=self.x_start_shape) * noise)
        return sample

    @tf.function(reduce_retracing=True)
    def p_losses(self, x_start, context_output, t):
        print("tracing check p_losses")
        noise = tf.random.normal(shape=self.x_start_shape)
        x_noisy = self.q_sample(x_start, t, noise)

        if self.denoise_fn.embd_type == "01":
            #tt = tf.expand_dims(tf.cast(t, tf.float32) , -1) / self.n_denoising_steps
            fx = self.denoise_fn(x_noisy, tf.expand_dims(tf.cast(t, tf.float32), -1) / self.n_denoising_steps_float, context_output)
        else:
            fx = self.denoise_fn(x_noisy, t, context_output)

        if self.pred_mode == "noise":
            if self.use_loss_weight:
                if self.loss_weight_min > 0:
                    weight = (self.train_snr[t].clamp(max=self.loss_weight_min) / self.train_snr[t])
                else:
                    weight = (self.train_snr[t].clamp(min=-self.loss_weight_min) / self.train_snr[t])
            else:
                weight = tf.ones(1)
            if self.loss_type == "l2":
                err = tf.reduce_mean(tf.square(noise - fx), axis=[1, 2, 3])
                err = tf.reduce_mean(err * weight)
            else:
                raise NotImplementedError()
        elif self.pred_mode == "x":
            if self.use_loss_weight:
                weight = tf.gather(self.train_snr,t, axis=-1)
                # weight = tf.clip_by_value(tf.gather(self.train_snr,t, axis=-1), 0,
                #                           self.loss_weight_min) if self.loss_weight_min > 0 else tf.clip_by_value(
                #     tf.gather(self.train_snr,t, axis=-1), self.loss_weight_min, 1e8)
            else:
                weight = tf.ones(1)
            if self.loss_type == "l1":
                err = tf.reduce_mean(tf.abs(x_start - fx), axis=[1, 2, 3])
                err = tf.reduce_mean(err * tf.sqrt(weight))
            elif self.loss_type == "l2":
                err = tf.reduce_mean(tf.square(x_start - fx), axis=[1, 2, 3])
                err = tf.reduce_mean(err * weight)
            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError()



        loss = err # self.lagrangian_beta * tf.reduce_mean(context_dict["bpp"]) + err

        return loss

    @tf.function(reduce_retracing=True)
    def call(self, images, input_snr = None):
        # B, C, H, W = self.img_batch_shape
        print("tracing check denoising_diffusion call")
        B = self.batch_size #tf.shape(images)[0]
        t = tf.random.uniform([B], minval=0, maxval=self.n_denoising_steps, dtype=tf.int32)

        ## train data does not contain freq data
        input_img, target_img, side_info = tf.split(images, num_or_size_splits=3, axis=-1)
        input_img = tf.squeeze(input_img, axis=-1)
        target_img = tf.squeeze(target_img, axis=-1)

        # center the data
        input_img = input_img * 2.0 - 1.0
        target_img = target_img * 2.0 - 1.0

        # context_dict_output, context_dict_bps, context_dict_q_latent, context_dict_a_loss = self.context_fn(input_img)# , training=True)

        # leave the uplink channel info as it is
        input_h = side_info # nearby uplink channel info used for simulating the uplink transmission
        
        if input_snr is None:
            input_snr = tf.random.uniform(shape=[B], minval=-10.0, maxval=10.0) # snr for uplink transmission
            context, context_dict_output, context_dict_bps, context_dict_q_latent = self.context_fn(input_img, input_h, input_snr)
        
        # @tf.function
        # def context_fn_wrapper():
        #     return self.context_fn(input_img, input_h, input_snr)
        
        # graph_info = profile(context_fn_wrapper.get_concrete_function().graph, 
        #                     options=ProfileOptionBuilder.float_operation())
        # context_flops = graph_info.total_float_ops
        
        if self.use_side_info is True:
            side_info = tf.squeeze(side_info, axis=-1)
            context_dict_output[0] = tf.concat([context_dict_output[0], side_info], axis=1)

        # ── compute per‑level p‑losses ─────────────────────────────────────────────
        per_level_losses = []
        loss = 0.0
        for i, out in enumerate(context_dict_output):
            
            # @tf.function
            # def p_losses_wrapper():
            #     return self.p_losses(target_img, out, t)
            
            # graph_info = profile(p_losses_wrapper.get_concrete_function().graph, 
            #                     options=ProfileOptionBuilder.float_operation())
            # p_losses_flops = graph_info.total_float_ops
            # print(f"p_losses FLOPs (level {i}): {p_losses_flops}")
            # print(f"context_fn FLOPs: {context_flops}")

            per_level_losses.append(self.p_losses(target_img, out, t))
            loss += self.mrl_weights[i] * per_level_losses[-1]

        loss = loss / tf.reduce_sum(self.mrl_weights)

        ## fix me
        # context_dict_output = [context_dict_output]

        # loss = self.p_losses(target_img, context_dict_output, t)#, aux_img=images)

        return loss #, context_dict_a_loss #self.get_extra_loss()

