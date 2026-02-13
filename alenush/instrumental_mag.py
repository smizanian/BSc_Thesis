import logging
import csv
import math
import numpy as np

from alenush.seeing import SeeingModel

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()

class InstrumentalMag:

    def __init__(self, image: np.ndarray):
        self.image: np.ndarray = image
        self.model = SeeingModel(image)

    def _calculate_radius(self):
        brightest_pixel = np.max(self.image)
        arr = cropped / brightest_pixel
        return self.model.get_fwhm(arr)

    def _calculate_sky_noise(self, radius):
        l = np.min(self.image.shape[0], self.image.shape[1])/2
        sky = []
        height, width = self.image.shape
        x_c, y_c = width//2, height//2
        for i in range(height):
            for j in range(width):
                distance = ((i - y_c)**2 + (j - x_c)**2)**0.5
                if (distance > radius and distance <= l):
                    sky.append(image[i][j])
        return np.median(sky)

    def _calculate_star_signal(self):
        radius = self._calculate_radius()
        sky = _calculate_sky_noise(radius)
        N = 0
        star = 0
        
        height, width = self.image.shape
        x_c, y_c = width//2, height//2
        for i in range(height):
            for j in range(width):
                distance = ((i - y_c)**2 + (j - x_c)**2)**0.5
                if (distance <= radius):
                    star += image[i][j]
                    N++
        return star - (N * sky)

    def instrumental_mag(self, t):
        star = self._calculate_star_signal()
        b = star / t
        mag = -2.5 * math.log10(b)
        return mag
    
