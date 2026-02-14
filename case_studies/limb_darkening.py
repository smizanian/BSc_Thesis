import os
import numpy as np
from typing import List
from matplotlib import pyplot as plt

from alenush.stacking import FITSHandler, DarkImage, FlatImage
from alenush.sun_basic import SunImage

fits_handler = FITSHandler()
fits_handler.set_local_path("/data/")
all_data = get_image_from_fits(21)
dark = all_data[0:9]
flat = all_data[10:19]
light = all_data[20]

exposure_times = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
dark_image = DarkImage(exposure_times, dark)
master_dark = dark_image.get_master_dark()
bias = dark_image.get_bias()
flat_image = FlatImage(flat)
master_flat = flat_image.get_normalized_flat(bias)
master_science = (light - master_dark) / master_flat

sun_image = SunImage(master_science)
x_c, y_c = sun_image.get_sun_center()
radius = sun_image.sun_radius()


sun_data_x = []
sun_data_y = []

for i in range(x_c):
    if(i%100 == 0):
        sun_data_x.append(x_c - i)
        sun_data_y.append(master_science[y_c][i])

sun_data_y = normalize_array(sun_data_y)

plt.plot(sun_data_x, sun_data_y)
plt.xlabel("radius (pixel)")
plt.ylabel("normalized flux")
plt.show()
radius = x_c - radius

mu_data_x = []
mu_data_y = []
edd_data_y = []
mu = 0
for i in range(x_c):
    if(i%100 == 0 and (x_c - i) < radius):
        mu = (1-((x_c - i)/radius)**2)**0.5
        mu_data_x.append(mu)
        mu_data_y.append(master_science[y_c][i])
        edd_data_y.append((2+3*mu)/5)

mu_data_y = normalize_array(mu_data_y)

plt.plot(mu_data_x, mu_data_y, color='g', label='solar data')
plt.plot(mu_data_x, edd_data_y, color='r', label='eddington modle')
plt.xlabel("mu")
plt.ylabel("normalized flux")
plt.show()
