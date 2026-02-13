import os
import logging
import numpy as np
from typing import List
from astropy.io import fits

logger = logging.getLogger()

class SunImage:

    def __init__(self, sun_image: np.ndarray):
        self._sun_image = sun_image
        self.x_c = -1
        self.y_c = -1

    def get_sun_center(self) -> tuple:
        sun_radius = 30
        start_h = 0
        end_h = 0
        row = self._sun_image[0]

        for i in range(len(row)):
            if(row[i]>sun_radius):
                start_h = i
                break

        for i in range(len(row)):
            if(row[len(row)-i]>sun_radius):
                end_h = i
                break

        start_v = 0        
        for i in range(len(self._sun_image)):
            if(images[16][i][len(row)]>sun_radius):
                start_v = i
                break;

        self.y_c = start_v + (self._sun_image) - start_v)//2
        self.x_c = start_h + (row) - start_h - end_h)//2
        center = (x_c, y_c)
        return center

    def sun_radius(self) -> int:
        sun_radius = 30
        x, y = x_c, y_c
        if self.x_c == -1 or self.y_c == -1:
            x, y = self.get_sun_center()
        radius = 0
        for i in range(x_c):
            if(self._sun_image[y_c][i]>sun_radius and radius==0):    
                radius = i
        return radius
        
