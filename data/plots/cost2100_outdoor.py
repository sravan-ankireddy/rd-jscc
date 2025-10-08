import matplotlib.pyplot as plt
from global_config import ROOT_DIRECTORY
import os

"""
Reference 1.
Sun, Xinran, Zhengming Zhang, and Luxi Yang. "An Efficient Network with Novel Quantization Designed for Massive MIMO CSI Feedback." arXiv preprint arXiv:2405.20068 (2024).
https://arxiv.org/abs/2405.20068


Reference 2.
Guo, Jiajia, et al. "Convolutional neural network-based multiple-rate compressive sensing for massive MIMO CSI feedback: Design, simulation, and analysis." IEEE Transactions on Wireless Communications 19.4 (2020): 2827-2840.
https://arxiv.org/pdf/1906.06007


Reference 3.
Ravula, Sriram, and Swayambhoo Jain. "Deep autoencoder-based massive MIMO CSI feedback with quantization and entropy coding." 2021 IEEE Global Communications Conference (GLOBECOM). IEEE, 2021.
https://ieeexplore.ieee.org/abstract/document/9685912
"""


data = {

    "CsiConformer UQ_E2E":      {(96, -1.69), (128, -2.18), (160, -2.13), (384, -3.21), (512, -5.80), (640, -6.52)},
    "CsiConformer muQ_E2E":     {(96, -2.11), (128, -2.84), (160, -2.80), (384, -4.62), (512, -6.27), (640, -6.87)},
    "CsiConformer base_VQ_VAE": {(96, -2.25), (128, -2.40), (160, -3.13), (384, -5.64), (512, -6.27), (640, -7.13)},
    "CsiConformer SVQ_VAE":     {(96, -2.93), (128, -3.01), (160, -3.44), (384, -6.69), (512, -6.91), (640, -7.40)},

    "CSINet+": {(384, -4.94), (738, -6.67), (1536, -9.96)},

    "Deep AE Entropy Coding":{
    (1024*1.649560117302053, -10.916086915829538),
    #(1024*1.7825024437927666, -11.50757315779622),
    #(1024*2.353372434017595, -12.278085418356525),
    #(1024*2.6251221896383186, -12.591423847257754),
    (1024*0.8597262952101662, -8.731593560866038),
    (1024*0.456989247311828, -6.933167467398764),
    (1024*0.23020527859237538, -5.718592063393023)},

    "Proposed":                 {(128, -7.52), (192, -8.70), (256, -9.51), (320, -10.86)}

}


# Plot Rate-Distortion Curves
plt.figure(figsize=(9, 4))

for algorithm, points in data.items():
    rates, distortions = zip(*sorted(points))  # Sort points by rate
    plt.plot(distortions, rates, marker='o', label=algorithm)

# Customize the plot
plt.title('Rate-Distortion Curves')
plt.ylabel('Rate (Bit)')
plt.xlabel('NMSE (dB)')
plt.legend()
plt.grid(True)

# Show the plot
#plt.show()
plt.savefig(os.path.join(ROOT_DIRECTORY, "rd_curves_cost2100.pdf"))
plt.show()
