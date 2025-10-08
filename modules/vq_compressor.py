import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, LeakyReLU,
    Activation, GlobalAveragePooling2D,
    Concatenate, Dense, Reshape, Add,
    Layer, Flatten
)
import numpy as np

# ── AF attention module as a Layer ─────────────────────────────────────────
class AFModule(Layer):
    def __init__(self, **kwargs):
        super(AFModule, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gap = GlobalAveragePooling2D()
        self.concat = Concatenate(axis=-1)
        self.dense1 = Dense(16, activation='relu')
        self.dense2 = Dense(input_shape[-1], activation='sigmoid')
        super(AFModule, self).build(input_shape)

    def call(self, inputs, snr):
        m = self.gap(inputs)
        m = self.concat([m, snr])
        m = self.dense1(m)
        m = self.dense2(m)
        m = tf.expand_dims(tf.expand_dims(m, 1), 1)
        return inputs * m

# ── AFCsiNetPlus encoder as a Layer ────────────────────────────────────────
class AFCsiNetPlusEncoderLayer(Layer):
    def __init__(self, encoded_dim, **kwargs):
        super(AFCsiNetPlusEncoderLayer, self).__init__(**kwargs)
        self.conv1 = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')
        self.bn1 = BatchNormalization()
        self.act1 = LeakyReLU()
        self.af1 = AFModule()

        self.conv2 = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')
        self.bn2 = BatchNormalization()
        self.act2 = LeakyReLU()
        self.af2 = AFModule()

        self.flatten = Flatten()
        self.dense  = Dense(encoded_dim, activation='tanh', kernel_initializer='truncated_normal', name='encoded')

    def call(self, x, snr):
        x = self.conv1(x); x = self.bn1(x); x = self.act1(x); x = self.af1(x, snr)
        x = self.conv2(x); x = self.bn2(x); x = self.act2(x); x = self.af2(x, snr)
        x = self.flatten(x)
        return self.dense(x)

# ── AFCsiNetPlus encoder as a Layer ────────────────────────────────────────
class AFCsiNetPlusEncoderLayerLarge(Layer):
    def __init__(self, encoded_dim, **kwargs):
        super(AFCsiNetPlusEncoderLayerLarge, self).__init__(**kwargs)
        self.conv1 = Conv2D(2, (11, 11), padding='same', kernel_initializer='truncated_normal')
        self.bn1 = BatchNormalization()
        self.act1 = LeakyReLU()
        self.af1 = AFModule()

        self.conv2 = Conv2D(32, (9, 9), padding='same', kernel_initializer='truncated_normal')
        self.bn2 = BatchNormalization()
        self.act2 = LeakyReLU()
        self.af2 = AFModule()

        self.conv3 = Conv2D(48, (7, 7), padding='same', kernel_initializer='truncated_normal')
        self.bn3 = BatchNormalization()
        self.act3 = LeakyReLU()
        self.af3 = AFModule()

        self.conv4 = Conv2D(2, (5, 5), padding='same', kernel_initializer='truncated_normal')
        self.bn4 = BatchNormalization()
        self.act4 = LeakyReLU()
        self.af4 = AFModule()

        self.flatten = Flatten()
        self.dense  = Dense(encoded_dim, activation='tanh', kernel_initializer='truncated_normal', name='encoded')

    def call(self, x, snr):
        x = self.conv1(x); x = self.bn1(x); x = self.act1(x); x = self.af1(x, snr)
        x = self.conv2(x); x = self.bn2(x); x = self.act2(x); x = self.af2(x, snr)
        x = self.conv3(x); x = self.bn3(x); x = self.act3(x); x = self.af3(x, snr)
        x = self.conv4(x); x = self.bn4(x); x = self.act4(x); x = self.af4(x, snr)
        x = self.flatten(x)
        return self.dense(x)

# ── AFCsiNetPlus decoder as a Layer ────────────────────────────────────────
class AFCsiNetPlusDecoderLayer(Layer):
    def __init__(self, residual_num=5, **kwargs):
        super(AFCsiNetPlusDecoderLayer, self).__init__(**kwargs)
        self.dense0      = Dense(32*32*2, activation='linear', kernel_initializer='truncated_normal', name='decoded')
        self.reshape0    = Reshape((32, 32, 2))

        self.af0    = AFModule()
        self.conv0  = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')
        self.bn0    = BatchNormalization()
        self.act0   = Activation('sigmoid')
        self.af0b   = AFModule()

        self.residual_num = residual_num
        self.res_blocks   = []
        for _ in range(residual_num):
            block = {}
            block['conv1'] = Conv2D(8, (7, 7), padding='same', kernel_initializer='truncated_normal')
            block['bn1']   = BatchNormalization()
            block['act1']  = LeakyReLU()
            block['af1']   = AFModule()

            block['conv2'] = Conv2D(16, (5, 5), padding='same', kernel_initializer='truncated_normal')
            block['bn2']   = BatchNormalization()
            block['act2']  = LeakyReLU()
            block['af2']   = AFModule()

            block['conv3'] = Conv2D(2, (3, 3), padding='same', kernel_initializer='truncated_normal')
            block['bn3']   = BatchNormalization()
            block['act3']  = Activation('tanh')
            block['af3']   = AFModule()

            self.res_blocks.append(block)

        self.final_act = Activation('relu')

    def call(self, encoded, snr):
        x = self.dense0(encoded)
        x = self.reshape0(x)
        x = self.af0(x, snr)
        x = self.conv0(x); x = self.bn0(x); x = self.act0(x); x = self.af0b(x, snr)
        intermediates = []
        for idx, block in enumerate(self.res_blocks):
            shortcut = x
            y = block['conv1'](x); y = block['bn1'](y); y = block['act1'](y); y = block['af1'](y, snr)
            if idx == self.residual_num - 1:
                intermediates.append(y)
            y = block['conv2'](y); y = block['bn2'](y); y = block['act2'](y); y = block['af2'](y, snr)
            if idx == self.residual_num - 1:
                intermediates.append(y)
            y = block['conv3'](y); y = block['bn3'](y); y = block['act3'](y); y = block['af3'](y, snr)
            x = Add()([shortcut, y])
            if idx == self.residual_num - 1:
                intermediates.append(x)

        return intermediates#self.final_act(x)

# ── AFCsiNetPlus decoder as a Layer ────────────────────────────────────────
class AFCsiNetPlusDecoderLayerLarge(Layer):
    def __init__(self, residual_num=5, **kwargs):
        super(AFCsiNetPlusDecoderLayerLarge, self).__init__(**kwargs)
        self.dense0      = Dense(32*32*2, activation='linear', kernel_initializer='truncated_normal', name='decoded')
        self.reshape0    = Reshape((32, 32, 2))

        self.af0    = AFModule()
        self.conv0  = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')
        self.bn0    = BatchNormalization()
        self.act0   = Activation('sigmoid')
        self.af0b   = AFModule()

        self.residual_num = residual_num
        self.res_blocks   = []
        for _ in range(residual_num):
            block = {}
            block['conv1'] = Conv2D(16, (11, 11), padding='same', kernel_initializer='truncated_normal')
            block['bn1']   = BatchNormalization()
            block['act1']  = LeakyReLU()
            block['af1']   = AFModule()

            block['conv2'] = Conv2D(24, (7, 7), padding='same', kernel_initializer='truncated_normal')
            block['bn2']   = BatchNormalization()
            block['act2']  = LeakyReLU()
            block['af2']   = AFModule()

            # block['conv3'] = Conv2D(16, (5, 5), padding='same', kernel_initializer='truncated_normal')
            # block['bn3']   = BatchNormalization()
            # block['act3']  = LeakyReLU()
            # block['af3']   = AFModule()

            block['conv4'] = Conv2D(2, (5, 5), padding='same', kernel_initializer='truncated_normal')
            block['bn4']   = BatchNormalization()
            block['act4']  = Activation('tanh')
            block['af4']   = AFModule()

            self.res_blocks.append(block)

        self.final_act = Activation('relu')

    def call(self, encoded, snr):
        x = self.dense0(encoded)
        x = self.reshape0(x)
        x = self.af0(x, snr)
        x = self.conv0(x); x = self.bn0(x); x = self.act0(x); x = self.af0b(x, snr)
        intermediates = []
        for idx, block in enumerate(self.res_blocks):
            shortcut = x
            y = block['conv1'](x); y = block['bn1'](y); y = block['act1'](y); y = block['af1'](y, snr)
            if idx == self.residual_num - 1:
                intermediates.append(y)

            y = block['conv2'](y); y = block['bn2'](y); y = block['act2'](y); y = block['af2'](y, snr)
            if idx == self.residual_num - 1:
                intermediates.append(y)

            # y = block['conv3'](y); y = block['bn3'](y); y = block['act3'](y); y = block['af3'](y, snr)
            # if idx == self.residual_num - 1:
            #     intermediates.append(y)

            y = block['conv4'](y); y = block['bn4'](y); y = block['act4'](y); y = block['af4'](y, snr)
            x = Add()([shortcut, y])
            if idx == self.residual_num - 1:
                intermediates.append(x)

        return intermediates #self.final_act(x)
    
# ── AFCsiNetPlus encoder as a Layer ────────────────────────────────────────
class AFCsiNetPlusEncoderLayerMed(Layer):
    def __init__(self, encoded_dim, **kwargs):
        super(AFCsiNetPlusEncoderLayerMed, self).__init__(**kwargs)
        self.conv1 = Conv2D(2, (11, 11), padding='same', kernel_initializer='truncated_normal')
        self.bn1 = BatchNormalization()
        self.act1 = LeakyReLU()
        self.af1 = AFModule()

        self.conv2 = Conv2D(16, (9, 9), padding='same', kernel_initializer='truncated_normal')
        self.bn2 = BatchNormalization()
        self.act2 = LeakyReLU()
        self.af2 = AFModule()

        self.conv3 = Conv2D(32, (7, 7), padding='same', kernel_initializer='truncated_normal')
        self.bn3 = BatchNormalization()
        self.act3 = LeakyReLU()
        self.af3 = AFModule()

        self.conv4 = Conv2D(2, (5, 5), padding='same', kernel_initializer='truncated_normal')
        self.bn4 = BatchNormalization()
        self.act4 = LeakyReLU()
        self.af4 = AFModule()

        self.flatten = Flatten()
        self.dense  = Dense(encoded_dim, activation='tanh', kernel_initializer='truncated_normal', name='encoded')

    def call(self, x, snr):
        x = self.conv1(x); x = self.bn1(x); x = self.act1(x); x = self.af1(x, snr)
        x = self.conv2(x); x = self.bn2(x); x = self.act2(x); x = self.af2(x, snr)
        x = self.conv3(x); x = self.bn3(x); x = self.act3(x); x = self.af3(x, snr)
        x = self.conv4(x); x = self.bn4(x); x = self.act4(x); x = self.af4(x, snr)
        x = self.flatten(x)
        return self.dense(x)
    
# ── AFCsiNetPlus decoder as a Layer ────────────────────────────────────────
class AFCsiNetPlusDecoderLayerMed(Layer):
    def __init__(self, residual_num=5, **kwargs):
        super(AFCsiNetPlusDecoderLayerMed, self).__init__(**kwargs)
        self.dense0      = Dense(32*32*2, activation='linear', kernel_initializer='truncated_normal', name='decoded')
        self.reshape0    = Reshape((32, 32, 2))

        self.af0    = AFModule()
        self.conv0  = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')
        self.bn0    = BatchNormalization()
        self.act0   = Activation('sigmoid')
        self.af0b   = AFModule()

        self.residual_num = residual_num
        self.res_blocks   = []
        for _ in range(residual_num):
            block = {}
            block['conv1'] = Conv2D(8, (11, 11), padding='same', kernel_initializer='truncated_normal')
            block['bn1']   = BatchNormalization()
            block['act1']  = LeakyReLU()
            block['af1']   = AFModule()

            block['conv2'] = Conv2D(16, (7, 7), padding='same', kernel_initializer='truncated_normal')
            block['bn2']   = BatchNormalization()
            block['act2']  = LeakyReLU()
            block['af2']   = AFModule()

            # block['conv3'] = Conv2D(16, (5, 5), padding='same', kernel_initializer='truncated_normal')
            # block['bn3']   = BatchNormalization()
            # block['act3']  = LeakyReLU()
            # block['af3']   = AFModule()

            block['conv4'] = Conv2D(2, (5, 5), padding='same', kernel_initializer='truncated_normal')
            block['bn4']   = BatchNormalization()
            block['act4']  = Activation('tanh')
            block['af4']   = AFModule()

            self.res_blocks.append(block)

        self.final_act = Activation('relu')

    def call(self, encoded, snr):
        x = self.dense0(encoded)
        x = self.reshape0(x)
        x = self.af0(x, snr)
        x = self.conv0(x); x = self.bn0(x); x = self.act0(x); x = self.af0b(x, snr)
        intermediates = []
        for idx, block in enumerate(self.res_blocks):
            shortcut = x
            y = block['conv1'](x); y = block['bn1'](y); y = block['act1'](y); y = block['af1'](y, snr)
            if idx == self.residual_num - 1:
                intermediates.append(y)

            y = block['conv2'](y); y = block['bn2'](y); y = block['act2'](y); y = block['af2'](y, snr)
            if idx == self.residual_num - 1:
                intermediates.append(y)

            # y = block['conv3'](y); y = block['bn3'](y); y = block['act3'](y); y = block['af3'](y, snr)
            # if idx == self.residual_num - 1:
            #     intermediates.append(y)

            y = block['conv4'](y); y = block['bn4'](y); y = block['act4'](y); y = block['af4'](y, snr)
            x = Add()([shortcut, y])
            if idx == self.residual_num - 1:
                intermediates.append(x)

        return intermediates #self.final_act(x)

# ── UplinkChannelMRC (unchanged) ────────────────────────────────────────────
class UplinkChannelMRC(Layer):
    def __init__(self, power_per_symbol=1, **kwargs):
        super(UplinkChannelMRC, self).__init__(**kwargs)
        self.power_per_symbol = power_per_symbol

    def call(self, features, h_real, snr_db):
        h_real = tf.squeeze(h_real, axis=-1)
        h_real = tf.transpose(h_real, (0, 2, 3, 1))
        inter_shape = tf.shape(features)

        f = Flatten()(features)
        dim_z = tf.shape(f)[1] // 2
        z_in = tf.complex(f[:, :dim_z], f[:, dim_z:])

        norm_factor = tf.reduce_sum(
            tf.math.real(z_in * tf.math.conj(z_in)), axis=1, keepdims=True
        )
        z_in_norm = z_in * tf.complex(
            tf.sqrt(tf.cast(dim_z, dtype=tf.float32) / norm_factor), 0.0
        )

        h = tf.complex(h_real[:, :, :, 0], h_real[:, :, :, 1])
        z_in_norm = z_in_norm[..., tf.newaxis]
        z_in_transmit = np.sqrt(self.power_per_symbol) * z_in_norm
        fd = tf.multiply(z_in_transmit, h)
        rv = self.add_noise(fd, snr_db)

        h_norm2 = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(h)), axis=2, keepdims=True))
        h_norm2_c = tf.complex(h_norm2, tf.zeros_like(h_norm2))
        w = h / h_norm2_c
        mrc = tf.reduce_sum(tf.multiply(rv, tf.math.conj(w)), axis=2)

        z2r = tf.concat([tf.math.real(mrc), tf.math.imag(mrc)], 1)
        out = tf.reshape(z2r, inter_shape)
        return out

    def add_noise(self, fd, snr_db):
        noise_std = tf.sqrt(10 ** (-snr_db / 10))
        noise_std = tf.complex(noise_std, 0.)
        noise_std = tf.reshape(noise_std, [-1, 1, 1])
        noise = tf.complex(
            tf.random.normal(tf.shape(fd), 0, 1 / np.sqrt(2)),
            tf.random.normal(tf.shape(fd), 0, 1 / np.sqrt(2))
        )
        return fd + noise_std * noise

# ── Compressor using the new AFCsiNetPlusLayer classes ─────────────────────
class Compressor(tf.keras.Model):
    def __init__(
        self,
        batch_size,
        n_embeddings=16,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        channels=3,
        out_channels=3,
        bandwidth=(32,24,16,8),
        **kwargs
    ):
        super(Compressor, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.bandwidth = bandwidth
        self.bps = tf.constant(np.log2(n_embeddings) * 8 * 8)

        # self.encoder_layer = AFCsiNetPlusEncoderLayer(encoded_dim=2 * bandwidth)
        # self.decoder_layer = AFCsiNetPlusDecoderLayer()

        self.encoder_layer = AFCsiNetPlusEncoderLayerLarge(encoded_dim=2 * bandwidth[0]) # use the first/largest bandwidth
        # self.decoder_layer = AFCsiNetPlusDecoderLayerLarge()

        # self.encoder_layer = AFCsiNetPlusEncoderLayerMed(encoded_dim=2 * bandwidth)
        self.decoder_layer = AFCsiNetPlusDecoderLayerMed()

    @tf.function(reduce_retracing=True)
    def encode(self, x, snr):
        x = tf.transpose(x, [0, 2, 3, 1])
        snr_ = tf.expand_dims(snr, 1)
        return self.encoder_layer(x, snr_)

    @tf.function(reduce_retracing=True)
    def decode(self, encoded, snr):
        snr_ = tf.expand_dims(snr, 1)
        x_list = self.decoder_layer(encoded, snr_)

        # x = tf.transpose(x, [0, 3, 1, 2])

        # transpose everything in x_list [0, 3, 1, 2]
        for i in range(len(x_list)):
            x_list[i] = tf.transpose(x_list[i], [0, 3, 1, 2])

        x_list = x_list[::-1]

        ae_out = x_list[0]

        full_context = x_list

        return ae_out, full_context

    @tf.function(reduce_retracing=True)
    def call(self, input, input_h, input_snr=None, bandwidth=None):
        if bandwidth is None:
            bandwidth = self.bandwidth
        # ——— 1) SNR preprocessing ——————————————————————
        B = tf.shape(input)[0]
        if input_snr is None:
            input_snr = tf.random.uniform([B], -10.0, 10.0)
        else:
            input_snr = tf.broadcast_to(tf.reshape(input_snr, [-1]), [B])

        # ——— 2) Encode to latent ————————————————————————
        latent = self.encode(input, input_snr)     # shape [B, full_dim]
        full_dim = latent.shape[1]

        # ——— 3) Instantiate MRC layer once ——————————————————
        mrc_layer = UplinkChannelMRC()

        # fix me 
        # bandwidth = [8]

        # ——— 4-6) Combined MRC path + decoding ————————————————
        dims = [2 * b for b in bandwidth]
        outputs = []
        outputs_list = []

        for d in dims:
            ch_req = d // 2
            
            # Slice latent
            z_slice = latent[:, :d]
            
            # MRC combining
            mrc_out = mrc_layer(z_slice, input_h[:, :, :ch_req], input_snr)
            
            # Pad to full dimension
            padded = tf.pad(mrc_out, [[0, 0], [0, full_dim - d]])
            
            # Decode
            out, out_list = self.decode(padded, input_snr)
            outputs.append(out)
            outputs_list.append(out_list)

        # ——— 7) Return all matryoshka outputs plus bitrate & latent ——
        return outputs, outputs_list, self.bps, latent

# ── VQCompressor inherits Compressor unchanged ───────────────────────────────
class VQCompressor(Compressor):
    def __init__(
        self,
        batch_size,
        n_embeddings=16,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        channels=3,
        out_channels=3,
        bandwidth=(32,24,16,8),
        **kwargs
    ):
        super(VQCompressor, self).__init__(
            batch_size=batch_size,
            n_embeddings=n_embeddings,
            dim=dim,
            dim_mults=dim_mults,
            reverse_dim_mults=reverse_dim_mults,
            channels=channels,
            out_channels=out_channels,
            bandwidth=bandwidth,
            **kwargs
        )

if __name__ == "__main__":
    batch = tf.ones((4, 2, 32, 32))
    h     = tf.ones((4, 2, 32, 32, 1))
    model = VQCompressor(batch_size=4)
    out, bps, lat = model(batch, h)
    print(out.shape, bps.numpy(), lat.shape)
