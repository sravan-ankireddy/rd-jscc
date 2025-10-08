import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Reshape, Dense, Add, Concatenate, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras import Model

# ConvBN Layer
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


# CRBlock Layer
class CRBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = tf.keras.Sequential([
            ConvBN(2, 7, 3),
            LeakyReLU(alpha=0.3),
            ConvBN(7, 7, [1, 3]),
            LeakyReLU(alpha=0.3),
            ConvBN(7, 7, [3, 1])
        ])
        self.path2 = tf.keras.Sequential([
            ConvBN(2, 7, [1, 5]),
            LeakyReLU(alpha=0.3),
            ConvBN(7, 7, [5, 1])
        ])
        self.conv1x1 = ConvBN(14, 2, 1)
        self.leaky_relu = LeakyReLU(alpha=0.3)

    def call(self, x):
        shortcut = x
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = tf.concat([out1, out2], axis=1)
        out = self.conv1x1(out)
        return self.leaky_relu(out + shortcut)


# Channel Pool Layer (max and mean pooling)
class ChannelPool(tf.keras.layers.Layer):
    def call(self, x):
        max_pool = tf.reduce_max(x, axis=1, keepdims=True)
        mean_pool = tf.reduce_mean(x, axis=1, keepdims=True)
        return tf.concat([max_pool, mean_pool], axis=1)


# SpatialGate Layer
class SpatialGate(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = ConvBN(2, 1, kernel_size)

    def call(self, x):
        x_compress = self.compress(x)
        scale = tf.sigmoid(self.spatial(x_compress))
        return x * scale


# SELayer (Squeeze and Excitation Layer)
class SELayer(tf.keras.layers.Layer):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = GlobalAveragePooling2D(data_format="channels_first")
        self.fc1 = Dense(channel // reduction, activation='relu', use_bias=False)
        self.fc2 = Dense(channel, activation='sigmoid', use_bias=False)

    def call(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x)
        y = tf.reshape(y, (-1, c))
        y = self.fc1(y)
        y = self.fc2(y)
        y = tf.reshape(y, (-1, c, 1, 1))
        return x * y


# Encoder Layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, reduction=4):
        super(EncoderLayer, self).__init__()
        total_size, in_channel, h, w = 2048, 2, 32, 32
        self.encoder1 = tf.keras.Sequential([
            ConvBN(in_channel, 2, 3),
            LeakyReLU(alpha=0.3),
            ConvBN(2, 2, (1, 9)),
            LeakyReLU(alpha=0.3),
            ConvBN(2, 2, (9, 1))
        ])
        self.encoder2 = ConvBN(in_channel, 32, 1)
        self.conv_final = tf.keras.Sequential([
            LeakyReLU(alpha=0.3),
            ConvBN(34, 2, 1),
            LeakyReLU(alpha=0.3)
        ])
        self.sa = SpatialGate()
        self.se = SELayer(32)
        self.fc = Dense(total_size // reduction, activation='linear')

    def call(self, x):
        encode1 = self.encoder1(x)
        encode1 = self.sa(encode1)
        encode2 = self.encoder2(x)
        encode2 = self.se(encode2)
        out = tf.concat([encode1, encode2], axis=1)
        out = self.conv_final(out)
        out = tf.reshape(out, (tf.shape(out)[0], -1))
        out = self.fc(out)
        return out


# Decoder Layer
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, reduction=4):
        super(DecoderLayer, self).__init__()
        total_size, in_channel, h, w = 2048, 2, 32, 32
        self.fc = Dense(total_size, activation='linear')
        self.decoder_conv = tf.keras.Sequential([
            ConvBN(2, 2, 5),
            LeakyReLU(alpha=0.3),
            CRBlock(),
            CRBlock()
        ])
        self.hsigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        out = self.fc(x)
        out = tf.reshape(out, (-1, in_channel, h, w))
        out = self.decoder_conv(out)
        out = self.hsigmoid(out)
        return out


# CLNet Model
class CLNet(Model):
    def __init__(self, reduction=4):
        super(CLNet, self).__init__()
        self.encoder = EncoderLayer(reduction=reduction)
        self.decoder = DecoderLayer(reduction=reduction)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
