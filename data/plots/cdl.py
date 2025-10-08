import matplotlib.pyplot as plt
from global_config import ROOT_DIRECTORY
import os

"""

"""


data = {
    "Proposed":        {(320, -11.01), (256, -10.26), (128, -9.02), (64, -5.98)},
    "CSINet module with uniform quantization":   {(256, -6.11), (192, -6.03), (132, -5.85), (64, -5.65)},
    "CRNet module with uniform quantization":   {(256, -6.11), (192, -6.03), (132, -5.85)},
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
plt.savefig(os.path.join(ROOT_DIRECTORY, "rd_curves_cdl.pdf"))
plt.show()
