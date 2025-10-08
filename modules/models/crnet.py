import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Dense, Reshape, Concatenate, Add, ZeroPadding2D
from tensorflow.keras import Model
from collections import OrderedDict

from modules.models.tanh_uniform_quantizer import NBitQuantizationLayer

class ConvBN(tf.keras.layers.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        super(ConvBN, self).__init__()

        # Compute padding manually based on kernel size
        if isinstance(kernel_size, (list, tuple)):
            padding = [(k - 1) // 2 for k in kernel_size]
            padding_height, padding_width = padding
        else:
            padding_height = padding_width = (kernel_size - 1) // 2

        # ZeroPadding2D applies the padding explicitly before the convolution
        self.padding = ZeroPadding2D(padding=(padding_height, padding_width), data_format='channels_first')

        # Define the convolution and batch normalization layers
        self.conv = Conv2D(filters=out_planes, kernel_size=kernel_size, strides=stride,
                           padding='valid', groups=groups, use_bias=False,
                           data_format='channels_first')  # No implicit padding in conv (set to 'valid')
        self.bn = BatchNormalization(axis=1)  # Batch normalization over channels (axis=1 for 'channels_first')

    def call(self, inputs):
        x = self.padding(inputs)  # Explicitly apply the padding
        x = self.conv(x)
        x = self.bn(x)
        return x


class CRBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = tf.keras.Sequential([
            ConvBN(2, 7, 3),
            LeakyReLU(alpha=0.3),
            ConvBN(7, 7, [1, 9]),
            LeakyReLU(alpha=0.3),
            ConvBN(7, 7, [9, 1])
        ])
        self.path2 = tf.keras.Sequential([
            ConvBN(2, 7, [1, 5]),
            LeakyReLU(alpha=0.3),
            ConvBN(7, 7, [5, 1])
        ])
        self.conv1x1 = ConvBN(14, 2, 1)
        self.relu = LeakyReLU(alpha=0.3)

    def call(self, x):
        identity = x

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = Concatenate(axis=1)([out1, out2])  # Concatenating along the channel axis
        out = self.relu(out)
        out = self.conv1x1(out)

        out = Add()([out, identity])
        out = self.relu(out)
        return out


class CRNet(Model):
    def __init__(self, batch_size, latent_dim, use_side_info):
        super(CRNet, self).__init__()
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.img_height = 32
        self.img_width = 32
        self.img_channels = 2
        self.img_total = 2048
        self.use_side_info = use_side_info
        self.n_quantization_bits = 6
        self.quantizer = NBitQuantizationLayer(n_bits=self.n_quantization_bits)

        # Encoder
        self.encoder1 = tf.keras.Sequential([
            ConvBN(self.img_channels, 2, 3),
            LeakyReLU(alpha=0.3),
            ConvBN(2, 2, [1, 9]),
            LeakyReLU(alpha=0.3),
            ConvBN(2, 2, [9, 1])
        ])
        self.encoder2 = ConvBN(self.img_channels, 2, 3)
        self.encoder_conv = tf.keras.Sequential([
            LeakyReLU(alpha=0.3),
            ConvBN(4, 2, 1),
            LeakyReLU(alpha=0.3)
        ])
        self.encoder_fc = Dense(self.latent_dim)

        # Decoder
        self.decoder_fc = Dense(self.img_total)
        self.decoder_feature = tf.keras.Sequential([
            ConvBN(2, 2, 5),
            LeakyReLU(alpha=0.3),
            CRBlock(),
            CRBlock()
        ])
        #self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.last_tanh = tf.keras.layers.Activation('tanh')


    @tf.function
    def call(self, images, training=None, mask=None):
        input_img, target_img, side_info = tf.split(images, num_or_size_splits=3, axis=-1)
        input_img = tf.squeeze(input_img, axis=-1)
        target_img = tf.squeeze(target_img, axis=-1)
        side_info = tf.squeeze(side_info, axis=-1)

        #output = self.model(input_img)
        # Encoder
        encode1 = self.encoder1(input_img)
        encode2 = self.encoder2(input_img)
        out = Concatenate(axis=1)([encode1, encode2])
        out = self.encoder_conv(out)
        out = self.encoder_fc(tf.reshape(out, (self.batch_size, -1)))
        out = self.quantizer(out)

        # Decoder
        out = self.decoder_fc(out)
        out = Reshape((self.img_channels, self.img_height, self.img_width))(out)

        if self.use_side_info is True:
            out = Concatenate(axis=1)([out, side_info])

        out = self.decoder_feature(out)
        #output = self.sigmoid(out)
        output = self.last_tanh(out)


        mse_loss = tf.reduce_mean(tf.square(target_img - output))  # MSE loss between output and target
        aux_loss = 0.0  # Add any auxiliary loss if needed
        return mse_loss, aux_loss


    @tf.function
    def predict(self, images, training=None, mask=None):
        input_img, target_img, side_info = tf.split(images, num_or_size_splits=3, axis=-1)
        input_img = tf.squeeze(input_img, axis=-1)
        side_info = tf.squeeze(side_info, axis=-1)

        #output = self.model(input_img)
        encode1 = self.encoder1(input_img)
        encode2 = self.encoder2(input_img)
        out = Concatenate(axis=1)([encode1, encode2])
        out = self.encoder_conv(out)
        out = self.encoder_fc(tf.reshape(out, (self.batch_size, -1)))
        out = self.quantizer(out)
        # Decoder
        out = self.decoder_fc(out)
        out = Reshape((self.img_channels, self.img_height, self.img_width))(out)
        if self.use_side_info is True:
            out = Concatenate(axis=1)([out, side_info])

        out = self.decoder_feature(out)
        #output = self.sigmoid(out)
        output = self.last_tanh(out)

        return output, self.latent_dim * self.n_quantization_bits

