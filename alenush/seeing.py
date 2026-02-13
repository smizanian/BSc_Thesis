import logging
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import List, Tuple, Dict
from mpl_toolkits import mplot3d

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()

class SeeingModel:

    def __init__(self, image: np.ndarray):
        self.image: np.ndarray = image

    def gaussian(x, amplitude, xo, yo, sigma_x, sigma_y, theta):
        y = x
        a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
        c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
        g = amplitude * np.exp(-(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
        return g.ravel()


    def get_fwhm(data: np.ndarray) -> float:
        max_val = np.max(data)
        half_max_val = max_val / 2
        indices = np.where(data >= half_max_val)
        x_indices = indices[0]
        y_indices = indices[1]
        fwhm_x = (x_indices.max() - x_indices.min()) * PIXEL_SIZE
        fwhm_y = (y_indices.max() - y_indices.min()) * PIXEL_SIZE
        return (fwhm_x + fwhm_y) / 2


    def get_fitted_data(arr: np.ndarray):
        x, y = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
        data = arr.ravel()
        xdata = np.vstack((x.ravel(), y.ravel()))
        initial_guess = [1, 10, 10, 1, 1, 0]
        popt, pcov = curve_fit(gaussian, xdata, data, p0=initial_guess)
        fitted_data = gaussian(xdata, *popt).reshape(arr.shape)
        return fitted_data


    def draw_3d_plot(data_1: np.ndarray, data_2: np.ndarray):
        x = np.arange(0, 21, 1)
        y = np.arange(0, 21, 1)
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        
        ax1.plot_surface(X, Y, data_1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Original Data')
        ax1.set_title('3D Plot of Original Data')

        ax2.plot_surface(X, Y, data_2)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Fitted Data')
        ax2.set_title('3D Plot of Fitted Data')

        plt.subplots_adjust(wspace=0.5)
        return fig
    

class StarClusterImage:
    
    STAR_FLUX_THRESHOLD_STD_DIST = 20
    STAR_PIXEL_COUNT_THRESHOLD = 30
    STAR_BORDER_THRESHOLD = 3
    STAR_WEIGHT_BORDER_THRESHOLD = 10
    STAR_DISTANCE_THRESHOLD = 5
    CROPPED_STAR_IMAGE_SIZE = 10

    def __init__(self, image: np.ndarray):
        self.image: np.ndarray = image
        self.median: float = np.median(image)
        self.std: float = np.std(image)
        self.star_flux_threshold: float = self.median + self.STAR_FLUX_THRESHOLD_STD_DIST * self.std
        self.marks: np.ndarray = np.zeros(shape=image.shape, dtype=int)
        self.star_centers: List[Tuple[int, int]] = []
        self.sorted_star_weights: Dict[Tuple[int, int], float] = {}

    def calc_sorted_star_weights(self):
        star_weights = {}
        for star_center in self.star_centers:
            star_weights[star_center] = self._get_star_weight(*star_center)
        self.sorted_star_weights = dict(sorted(star_weights.items(), key=lambda item: item[1], reverse=True))

    def _get_star_weight(self, x: int, y: int) -> float:
        weight = 0
        for i in range(-1 * self.STAR_WEIGHT_BORDER_THRESHOLD, self.STAR_WEIGHT_BORDER_THRESHOLD + 1):
            for j in range(-1 * self.STAR_WEIGHT_BORDER_THRESHOLD, self.STAR_WEIGHT_BORDER_THRESHOLD + 1):
                weight += self.image[x + i][y + j]
        return weight

    def get_brightest_star_centers(self, count: int) -> List[Tuple[int, int]]:
        brightest_centers = []
        cnt = 0
        for star_center in self.sorted_star_weights.keys():
            brightest_centers.append(star_center)
            cnt += 1
            if cnt >= count:
                return brightest_centers
        return brightest_centers

    def find_image_centers(self):
        height, width = self.image.shape
        for i in range(height):
            for j in range(width):
                if self.image[i][j] > self.star_flux_threshold and self.marks[i][j] == 0:
                    star_center, pixel_count = self._get_star_center(x=i, y=j, x_range=(i, i), y_range=(j, j), star_pixel_count=0)
                    if pixel_count >= self.STAR_PIXEL_COUNT_THRESHOLD:
                        logger.info(f'Found star at {star_center} with size {pixel_count} pixels')
                        self.star_centers.append(star_center)
        self._remove_duplicate_centers()
        logger.info(f'Found {len(self.star_centers)} stars')

    def _remove_duplicate_centers(self):
        self.star_centers = list(set(self.star_centers))
        indexes_to_remove_set = set()
        for i in range(len(self.star_centers)):
            for j in range(i + 1, len(self.star_centers)):
                if self._is_single_star(self.star_centers[i], self.star_centers[j]):
                    indexes_to_remove_set.add(j)
        self.star_centers = [i for j, i in enumerate(self.star_centers) if j not in indexes_to_remove_set]

    @staticmethod
    def _is_single_star(center_1: Tuple[int, int], center_2: Tuple[int, int]):
        delta_x = center_1[0] - center_2[0]
        delta_y = center_1[1] - center_2[1]
        dist_2 = pow(delta_x, 2) + pow(delta_y, 2)
        return dist_2 <= pow(StarClusterImage.STAR_DISTANCE_THRESHOLD, 2)

    def _get_star_center(self, x: int, y: int, x_range: Tuple[int, int], y_range: Tuple[int, int],star_pixel_count: int) -> Tuple[Tuple[int, int], int]:

        self.marks[x][y] = 1

        for i in range(-1 * self.STAR_BORDER_THRESHOLD, self.STAR_BORDER_THRESHOLD):
            for j in range(-1 * self.STAR_BORDER_THRESHOLD, self.STAR_BORDER_THRESHOLD):
                if self._is_unvisited_and_star(x + i, y + j):
                    star_pixel_count += 1
                    if x + i < x_range[0]:
                        new_x_range = (x + i, x_range[1])
                        return self._get_star_center(x, y, new_x_range, y_range, star_pixel_count)
                    if x + i > x_range[1]:
                        new_x_range = (x_range[0], x + i)
                        return self._get_star_center(x, y, new_x_range, y_range, star_pixel_count)
                    if y + j < y_range[0]:
                        new_y_range = (y + j, y_range[1])
                        return self._get_star_center(x, y, x_range, new_y_range, star_pixel_count)
                    if y + j > y_range[1]:
                        new_y_range = (y_range[0], y + j)
                        return self._get_star_center(x, y, x_range, new_y_range, star_pixel_count)

        return self._find_center(x_range, y_range), star_pixel_count

    def _find_center(self,x_range: Tuple[int, int], y_range: Tuple[int, int]) -> Tuple[int, int]:
        sum_flux = 0
        sum_x, sum_y = 0, 0
        for i in range(x_range[0], x_range[1] + 1):
            for j in range(y_range[0], y_range[1] + 1):
                sum_flux += self.image[i][j]
                sum_x += i * self.image[i][j]
                sum_y += j * self.image[i][j]
        return int(sum_x / sum_flux), int(sum_y / sum_flux)

    def _is_unvisited_and_star(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or \ x >= self.image.shape[0] or \ y >= self.image.shape[1]:
            return False
        if self.marks[x][y]:
            return False
        if self.image[x][y] > self.star_flux_threshold:
            return True

    def get_cropped_image_by_center(self, star_center: Tuple[int, int]) -> np.ndarray:
        x_start = star_center[0] - self.CROPPED_STAR_IMAGE_SIZE
        x_end = star_center[0] + self.CROPPED_STAR_IMAGE_SIZE + 1
        y_start = star_center[1] - self.CROPPED_STAR_IMAGE_SIZE
        y_end = star_center[1] + self.CROPPED_STAR_IMAGE_SIZE + 1
        return self.image[x_start: x_end, y_start:y_end]
