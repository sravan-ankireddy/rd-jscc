import tensorflow as tf
import os
import numpy as np
from sionna.channel.tr38901 import AntennaArray, CDL
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
from sionna.utils import expand_to_rank
import warnings
from sionna import PI
from dataclasses import dataclass, field


@dataclass
class CDLChannelGenerator:
    """
        The goal of this dataclass is to generate a batch which consists of three different channel state information
        (CSI) instances.

        Consider time $t$ and corresponding CSI $H_t$. The goal is to generate a future CSI at time step $t+i$,
        $H_{t+i}$ where $i$ is the CSI feedback interval, typically 5ms.

    """
    batch_size: int
    max_ue_speed: float = 5
    num_ut: int = 1
    num_ut_ant: int = 1
    num_bs: int = 1
    num_bs_ant: int = 32
    num_subcarriers: int = 667
    subcarrier_spacing: float = 15e3
    num_ofdm_symbols_per_slot: int = 14
    ebno_db: list = field(default_factory=lambda: list(np.arange(-4, 11, 1.0)))
    cdl_model: str = 'C'
    delay_spread: float = 300e-9
    domain: str = "freq"
    direction: str = "downlink"
    ul_center_frequency: float = 1.91e9
    dl_center_frequency: float = 2.11e9
    perfect_csi: bool = False
    cyclic_prefix_length: int = 0
    pilot_ofdm_symbol_indices: list = field(default_factory=lambda: [2, 11])
    ber: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.ofdm_symbol_duration = (1. + self.cyclic_prefix_length/self.num_subcarriers) / self.subcarrier_spacing
        self.carrier_frequency = (self.ul_center_frequency + self.dl_center_frequency)/2.

        ut_array = AntennaArray(num_rows=1,
                                num_cols=1,
                                polarization="single",
                                polarization_type="V",
                                antenna_pattern="omni",
                                carrier_frequency=self.carrier_frequency)

        bs_array = AntennaArray(num_rows=1,
                                num_cols=self.num_bs_ant,  # / 2),
                                polarization="single",  # "dual",
                                polarization_type="V",  # "cross",
                                antenna_pattern="38.901",
                                carrier_frequency=self.carrier_frequency)

        self.cdl = CDL(self.cdl_model, self.delay_spread, self.carrier_frequency, ut_array, bs_array, self.direction,
                       min_speed=self.max_ue_speed, max_speed=self.max_ue_speed)


        total_num_subcarriers = (self.dl_center_frequency - self.ul_center_frequency + self.num_subcarriers * self.subcarrier_spacing) // self.subcarrier_spacing
        frequencies = subcarrier_frequencies(num_subcarriers=total_num_subcarriers, subcarrier_spacing=self.subcarrier_spacing)
        self.ul_frequencies = frequencies[:self.num_subcarriers]
        self.dl_frequencies = frequencies[-self.num_subcarriers:]

        check_ul_frequencies = self.ul_frequencies.numpy()
        check_dl_frequencies = self.dl_frequencies.numpy()

        abs_ul_frequencies = self.ul_frequencies + self.carrier_frequency
        abs_ul_center_frequency = np.median(abs_ul_frequencies)
        abs_dl_frequencies = self.dl_frequencies + self.carrier_frequency
        abs_dl_center_frequency = np.median(abs_dl_frequencies)

        print("hello")


    def get_batch(self):

        a, tau = self.cdl(self.batch_size,
                          num_time_steps=self.num_ofdm_symbols_per_slot * 1 + 1,
                          sampling_frequency=1/self.ofdm_symbol_duration)

        modified_a = tf.stack([a[:, :, :, :, :, :, 0], a[:, :, :, :, :, :, -1]], axis=-1)
        h_ul = cir_to_ofdm_channel(self.ul_frequencies, modified_a, tau, normalize=False)
        h_dl = cir_to_ofdm_channel(self.dl_frequencies, modified_a, tau, normalize=False)
        h_ul_t_1 = tf.squeeze(h_ul[:, :, :, :, :, -1, :])
        h_dl_t_1 = tf.squeeze(h_dl[:, :, :, :, :, -1, :])
        h_dl_t = tf.squeeze(h_dl[:, :, :, :, :, 0, :])

        # current_frobenius_norm = np.maximum(np.max(h_ul_t_1.real), np.max(h_ul_t_1.imag))

        h_ul_t_1 = tf.signal.ifft2d(h_ul_t_1)  # inner-most dimension Inverse FFT
        h_dl_t_1 = tf.signal.ifft2d(h_dl_t_1)  # inner-most dimension Inverse FFT
        h_dl_t = tf.signal.ifft2d(h_dl_t)  # inner-most dimension Inverse FFT
        h_ul_t_1 = h_ul_t_1[:, :, :32]
        h_dl_t_1 = h_dl_t_1[:, :, :32]
        h_dl_t = h_dl_t[:, :, :32]

        h_ul_t_1 = tf.stack([tf.math.real(h_ul_t_1), tf.math.imag(h_ul_t_1)], axis=-1)
        h_dl_t_1 = tf.stack([tf.math.real(h_dl_t_1), tf.math.imag(h_dl_t_1)], axis=-1)
        h_dl_t = tf.stack([tf.math.real(h_dl_t), tf.math.imag(h_dl_t)], axis=-1)

        h_ul_t_1 = h_ul_t_1 / tf.reduce_max(tf.abs(h_ul_t_1), axis=(1, 2, 3), keepdims=True)
        h_dl_t_1 = h_dl_t_1 / tf.reduce_max(tf.abs(h_dl_t_1), axis=(1, 2, 3), keepdims=True)
        h_dl_t = h_dl_t / tf.reduce_max(tf.abs(h_dl_t), axis=(1, 2, 3), keepdims=True)

        h_ul_t_1 =  (h_ul_t_1 + 1.0) / 2.0
        h_dl_t_1 =  (h_dl_t_1 + 1.0) / 2.0
        h_dl_t =    (h_dl_t + 1.0) / 2.0

        data_batch = tf.stack([h_dl_t, h_dl_t_1, h_ul_t_1], axis=-1)
        data_batch = tf.transpose(data_batch, [0,3,1,2,4])

        return data_batch

    def get_SF_batch(self):
        a, tau = self.cdl(self.batch_size,
                          num_time_steps=self.num_ofdm_symbols_per_slot * 1 + 1,
                          sampling_frequency=1/self.ofdm_symbol_duration)
        modified_a = tf.stack([a[:, :, :, :, :, :, 0], a[:, :, :, :, :, :, -1]], axis=-1)
        h_ul = cir_to_ofdm_channel(self.ul_frequencies, modified_a, tau, normalize=False)
        h_dl = cir_to_ofdm_channel(self.dl_frequencies, modified_a, tau, normalize=False)
        h_ul_t_1 = tf.squeeze(h_ul[:, :, :, :, :, -1, :])
        h_dl_t_1 = tf.squeeze(h_dl[:, :, :, :, :, -1, :])
        h_dl_t = tf.squeeze(h_dl[:, :, :, :, :, 0, :])

        h_ul_t_1 = tf.stack([tf.math.real(h_ul_t_1), tf.math.imag(h_ul_t_1)], axis=-1)
        h_dl_t_1 = tf.stack([tf.math.real(h_dl_t_1), tf.math.imag(h_dl_t_1)], axis=-1)
        h_dl_t = tf.stack([tf.math.real(h_dl_t), tf.math.imag(h_dl_t)], axis=-1)

        data_batch = tf.stack([h_dl_t, h_dl_t_1, h_ul_t_1], axis=-1)
        return data_batch

def visualize_channel(sample, v_min, v_max, domain="SF", name=""):
    import matplotlib.pyplot as plt
    import matplotlib
    # matplotlib.rc('text', usetex=True)
    #matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    if domain == "AD":
        zeropadded_sample = np.concatenate(
            [sample, np.zeros(shape=(32, 667 - 32))],
            axis=1)
        sample = np.fft.fft(zeropadded_sample, axis=1)
        sample = np.fft.fft(sample, axis=0)

    remaining_delay_components = 32
    n_tx_antenna, n_subcarriers = sample.shape
    sample_delay_domain = np.fft.ifft(sample, axis=1)
    sample_delay_domain = np.fft.ifft(sample_delay_domain, axis=0)
    sample_delay_domain_cropped = sample_delay_domain[:, :remaining_delay_components]
    sample_delay_domain_cropped_zp = np.concatenate([sample_delay_domain_cropped, np.zeros(shape=(n_tx_antenna, n_subcarriers - remaining_delay_components))], axis=1)
    sample_delay_domain_cropped_zp = np.fft.fft(sample_delay_domain_cropped_zp, axis=0)
    sample_recovered_sf = np.fft.fft(sample_delay_domain_cropped_zp, axis=1)


    colormap =  plt.get_cmap('gist_gray')

    import matplotlib.colors as colors
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    colormap = truncate_colormap(colormap, 0.25, 1.0)

    v_min = 0.0
    v_max = 1.2
    # Create two subplots
    plt.figure(figsize=(4, 10))  # Adjust the figure size as needed

    print((np.abs(sample)).max())
    # Subplot 1
    plt.subplot(4, 1, 1)  # 1 row, 2 columns, and this is the first subplot
    plt.imshow( (np.abs(sample)), cmap=colormap, interpolation='nearest', aspect='auto')#, vmin=v_min, vmax=v_max)  # Customize the colormap and interpolation
    plt.title(name+', Spatial/Freq (Y/X)')

    # Subplot 2
    plt.subplot(4, 1, 2)  # 1 row, 2 columns, and this is the second subplot
    plt.imshow( (np.abs(sample_delay_domain_cropped)), cmap=colormap, interpolation='nearest', aspect='auto')# , vmin=v_min, vmax=v_max)
    plt.title('Angular/Delay-cropped (Y/X)')

    # Subplot 3
    plt.subplot(4, 1, 3)  # 1 row, 2 columns, and this is the second subplot
    plt.imshow( (np.abs(sample_delay_domain)), cmap=colormap, interpolation='nearest', aspect='auto')#, vmin=v_min, vmax=v_max)
    plt.title('Angular/Delay (Y/X)')

    # Subplot 4
    plt.subplot(4, 1, 4)  # 1 row, 2 columns, and this is the second subplot
    plt.imshow(np.abs(sample_recovered_sf), cmap=colormap, interpolation='nearest', aspect='auto')#, vmin=v_min, vmax=v_max)
    plt.title('Spatial/Freq-recovered (Y/X)')
    plt.tight_layout()  # Ensure subplots don't overlap

    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(dir_path, name+"sample.pdf"))
    plt.show()

    return

if __name__ == "__main__":
    import random
    #random_seed = 2
    random_seed = 3
    random.seed(random_seed)
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    batch_size = 100
    channel_generator = CDLChannelGenerator(batch_size=batch_size,
                                            max_ue_speed=5,
                                            dl_center_frequency=1.91e9 + 667*15000,
                                            ul_center_frequency=1.91e9,)

    minibatch = channel_generator.get_batch()
    domain = "AD"
    variance = np.var(minibatch.numpy())

    for data_idx in range(1):
        single_sample = minibatch[data_idx].numpy() - 0.5
        v_min, v_max = 0, 5.25

        sample_input_source, sample_label, side_info = single_sample[:,:,:,0], single_sample[:,:,:,1], single_sample[:,:,:,2]
        sample_input_source = np.transpose(sample_input_source, [1,2,0])
        sample_label = np.transpose(sample_label, [1,2,0])
        side_info = np.transpose(side_info, [1,2,0])


        visualize_channel(sample_input_source[:,:,0] + 1j* sample_input_source[:,:,1], domain=domain, name="${H}^{(t)}_{DL}$", v_min=v_min, v_max=v_max)
        visualize_channel(sample_label[:,:,0] + 1j* sample_label[:,:,1], domain=domain, name="${H}^{(t+1)}_{UL}$", v_min=v_min, v_max=v_max)
        visualize_channel(side_info[:,:,0] + 1j* side_info[:,:,1], domain=domain, name="${H}^{(t+1)}_{DL}$", v_min=v_min, v_max=v_max)
        print(data_idx, " get instance")
