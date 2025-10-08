import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Reshape, Dense, LeakyReLU, Add, Concatenate
from tensorflow.keras import Model

from modules.models.tanh_uniform_quantizer import NBitQuantizationLayer


# Encoder Layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, latent_dim, img_total):
        super(EncoderLayer, self).__init__()
        self.img_total = img_total
        self.latent_dim = latent_dim

        # Define the encoder layers
        self.conv1 = Conv2D(2, (3, 3), padding='same', data_format="channels_first")
        self.bn1 = BatchNormalization(axis=1)
        self.fc = Dense(self.latent_dim, activation='linear')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Reshape((self.img_total,))(x)
        encoded = self.fc(x)
        return encoded


# Residual Block Layer
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(8, (3, 3), padding='same', data_format="channels_first")
        self.conv2 = Conv2D(16, (3, 3), padding='same', data_format="channels_first")
        self.conv3 = Conv2D(2, (3, 3), padding='same', data_format="channels_first")
        self.bn1 = BatchNormalization(axis=1)
        self.bn2 = BatchNormalization(axis=1)
        self.bn3 = BatchNormalization(axis=1)
        self.leaky_relu = LeakyReLU(alpha=0.3)

    def call(self, inputs, side_info=None):
        shortcut = inputs

        if side_info is not None:
            inputs = Concatenate(axis=1)([inputs, side_info])

        # First conv layer
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        # Second conv layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        # Third conv layer
        x = self.conv3(x)
        x = self.bn3(x)

        # Add the shortcut (skip connection)
        x = Add()([x, shortcut])
        x = self.leaky_relu(x)
        return x


# Decoder Layer
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, img_total, img_channels, img_height, img_width, residual_num, use_side_info):
        super(DecoderLayer, self).__init__()
        self.img_total = img_total
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        self.use_side_info = use_side_info

        # Define decoder layers
        self.fc = Dense(self.img_total, activation='linear')
        self.residual_blocks = [ResidualBlock() for _ in range(residual_num)]
        self.conv_final = Conv2D(2, (3, 3), activation='tanh', padding='same', data_format="channels_first")
        # self.conv_final = Conv2D(2, (3, 3), activation='tanh', padding='same', data_format="channels_first")

        if self.use_side_info is True:
            self.side_info_residual_block = ResidualBlock()
            self.conv_aux = Conv2D(2, (2, 2), padding='same', data_format="channels_first")

    def call(self, encoded, side_info=None):
        x = self.fc(encoded)
        x = Reshape((self.img_channels, self.img_height, self.img_width))(x)

        # Optionally concatenate side information

        # side_info = Reshape((1, self.img_height, self.img_width))(side_info)
        # side_info = self.side_info_residual_block(side_info)
        # x = Concatenate(axis=1)([x, side_info])
        # x = self.conv_aux(x)

        # Apply residual blocks
        for residual_block in self.residual_blocks:
            if self.use_side_info is True:
                x = residual_block(x, side_info)
            else:
                x = residual_block(x)

        decoded = self.conv_final(x)
        return decoded


# Main CSINet Model
class CSINet(Model):
    def __init__(self, batch_size, latent_dim, use_side_info):
        super(CSINet, self).__init__()
        self.img_height = 32
        self.img_width = 32
        self.img_channels = 2
        self.img_total = self.img_height * self.img_width * self.img_channels
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.use_side_info = use_side_info
        self.residual_num = 2
        self.n_quantization_bits = 6
        self.use_quantization = True

        # Define encoder and decoder layers
        self.encoder = EncoderLayer(latent_dim=self.latent_dim, img_total=self.img_total)

        if self.use_quantization is True:
            self.quantizer = NBitQuantizationLayer(n_bits=self.n_quantization_bits)

        self.decoder = DecoderLayer(img_total=self.img_total, img_channels=self.img_channels,
                                    img_height=self.img_height, img_width=self.img_width,
                                    residual_num=self.residual_num, use_side_info=self.use_side_info)

    @tf.function
    def call(self, images, training=None, mask=None):
        # Split the input tensor into input image, target image, and side info
        input_img, target_img, side_info = tf.split(images, num_or_size_splits=3, axis=-1)
        input_img = tf.squeeze(input_img, axis=-1)
        target_img = tf.squeeze(target_img, axis=-1)
        side_info = tf.squeeze(side_info, axis=-1)

        # Encoding process
        encoded = self.encoder(input_img)
        if self.use_quantization is True:
            encoded = self.quantizer(encoded)

        # Decoding process
        if self.use_side_info:
            decoded = self.decoder(encoded, side_info=side_info)
        else:
            decoded = self.decoder(encoded)

        # Compute Mean Squared Error (MSE) loss
        mse_loss = tf.reduce_mean(tf.square(target_img - decoded))
        aux_loss = 0.0  # Add any auxiliary loss if needed

        return mse_loss, aux_loss

    @tf.function
    def predict(self, images, training=None, mask=None):
        # Split the input tensor into input image, target image, and side info
        input_img, target_img, side_info = tf.split(images, num_or_size_splits=3, axis=-1)
        input_img = tf.squeeze(input_img, axis=-1)
        side_info = tf.squeeze(side_info, axis=-1)

        encoded = self.encoder(input_img)
        if self.use_quantization is True:
            encoded = self.quantizer(encoded)

        # Encoding and decoding for prediction
        if self.use_side_info:
            decoded = self.decoder(encoded, side_info=side_info)
        else:
            decoded = self.decoder(encoded)

        return decoded, self.latent_dim * self.n_quantization_bits
