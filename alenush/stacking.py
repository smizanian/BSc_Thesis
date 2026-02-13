import os
import logging
import numpy as np
from typing import List
from astropy.io import fits

from alenush.instrumental_mag import StarClusterImage

logger = logging.getLogger()


class FITSHandler:

    PATH = os.getcwd() + "/data/fits/"

    @classmethod
    def get_grayscale(cls, rgb_image: np.ndarray) -> np.ndarray:
        return np.dot(rgb_image.T, [0.2989, 0.5870, 0.1140])

    @classmethod
    def crop_image(cls, image: np.ndarray) -> np.ndarray:
        for i in range(4):
            image = np.delete(image, i, 0)
            image = np.delete(image, i, 1)
            image = np.delete(image, image.shape[0] - (i + 1), 0)
            image = np.delete(image, image.shape[1] - (i + 1), 1)
        return image

    def set_path(self, new_path: str):
        self.PATH = new_path

    def set_local_path(self, new_path: str):
        self.PATH = os.getcwd() + new_path

    def get_images_from_fits(self, count: int) -> List[np.ndarray]:
        filenames = os.listdir(self.PATH)
        filenames.sort()
        images = []
        logger.info('Reading FITS files...')
        for i in range(count):
            filename = filenames[i]
            if not filename.startswith('DSC'):
                continue
            image = self.get_image_from_fits_by_name(filename)
            images.append(image)
            logger.info(f'Done {filename}')
        return images

    def get_image_from_fits_by_name(self, filename: str) -> np.ndarray:
        logger.info(f'Reading {filename}')
        hdu_list = fits.open(self.PATH + filename)
        rgb_image = hdu_list[0].data
        hdu_list.close()
        logger.info(f'Processing {filename}')
        grayscale_image = self.get_grayscale(rgb_image)
        image = self.crop_image(grayscale_image)
        return image


class DarkImage:

    def __init__(self, exposure_time: str, images: List[np.ndarray]):
        self._exposure_time: str = exposure_time
        self._images: List[np.ndarray] = images
        self._stacked_images: np.ndarray = np.stack(images)
        self._median_array: np.ndarray = self._get_median_array(self._stacked_images)
        self._std_array: np.ndarray = self._get_std_array(self._stacked_images)

    @classmethod
    def _get_median_array(cls, stacked_images: np.ndarray) -> np.ndarray:
        return np.median(stacked_images, axis=0)

    @classmethod
    def _get_std_array(cls, stacked_images: np.ndarray) -> np.ndarray:
        return np.std(stacked_images, axis=0)

    def get_exposure_values_in_std_range(self, std_count: int) -> List[float]:
        values_in_std_range = []
        for j in range(self._stacked_images.shape[1]):
            for k in range(self._stacked_images.shape[2]):
                for i in range(self._stacked_images.shape[0]):
                    pixel_value = self._stacked_images[i][j][k]
                    pixel_median = self._median_array[j][k]
                    pixel_std = self._std_array[j][k]
                    if (pixel_median + std_count * pixel_std) > pixel_value > (pixel_median - std_count * pixel_std):
                        values_in_std_range.append(pixel_value)
        return values_in_std_range

    def get_dead_pixels(self) -> List[tuple]:
        dead_pixels = []
        for j in range(self._stacked_images.shape[1]):
            for k in range(self._stacked_images.shape[2]):
                for i in range(self._stacked_images.shape[0]):
                    pixel_value = self._stacked_images[i][j][k]
                    pixel_median = self._median_array[j][k]
                    if (10 > pixel_value >= 0):
                        index = (j, k)
                        dead_pixels.append(index)
        return dead_pixels

    def get_saturated_pixels(self) -> List[tuple]:
        saturated_pixels = []
        saturation_value = np.max(self._stacked_images)
        for j in range(self._stacked_images.shape[1]):
            for k in range(self._stacked_images.shape[2]):
                for i in range(self._stacked_images.shape[0]):
                    pixel_value = self._stacked_images[i][j][k]
                    pixel_median = self._median_array[j][k]
                    if (pixel_value = saturation_value):
                        index = (j, k)
                        dead_pixels.append(index)
        return dead_pixels

    def get_master_dark(self) -> np.ndarray:
        return np.median(self._stacked_images, axis=0)
    

    def get_bias(self) -> np.ndarray:
        minimum_exposure = np.argmin(self._exoosure_time)
        return self._stacked_images(minimum_exposure)


class FlatImage:
    
    def __init__(self, flat_images: List[np.ndarray]):
        stacked_image = np.stack(flat_images)
        self.image: np.ndarray = np.median(stacked_image, axis=0)

    def get_normalized_flat(self, flat_dark: np.ndarray):
        dark_cleaned_flat = self.image - flat_dark
        normalized_flat = dark_cleaned_flat / np.max(dark_cleaned_flat)
        return normalized_flat

class ScienceImage:

    def __init__(self, light_frames: List[np.ndarray], flat_image: np.ndarray, science_dark: np.ndarray):
        self.aligned_image: np.ndarray = None
        self.image_shape: Tuple = light_frames[0].shape
        self.science_images: List[np.ndarray] = []
        for img in light_frames:
            clean_image = (img - science_dark) / flat_image
            self.science_images.append(clean_image)

    def get_aligned_science(self) -> np.ndarray:
        if self.aligned_image = None:
            self.calculate_aligned_science()
        return self.aligned_image

    def calculate_aligned_science(self) -> np.ndarray:
        star_centers: List[Tuple[int, int]] = []
        
        for image in self.science_images:
            star_cluster_image = StarClusterImage(image)
            star_cluster_image.find_image_centers()
            star_cluster_image.calc_sorted_star_weights()
            brightest_star_center = star_cluster_image.get_brightest_star_centers(1)[0]
            star_centers.append(brightest_star_center)
            
        aligned_science = self.science_images[0].copy()
        maximum_delta_x, maximum_delta_y = 0, 0
        base_x, base_y = star_centers[0]
        
        for i in range(1, len(self.science_images)):
            img = self.science_images[i].copy()
            delta_x = star_centers[i][0] - base_x
            delta_y = star_centers[i][1] - base_y
            img = np.roll(img, -delta_y, axis=0)
            img = np.roll(img, -delta_x, axis=1)
            aligned_science += img
            if delta_x > maximum_delta_x:
                maximum_delta_x = delta_x
            if delta_y > maximum_delta_y:
                maximum_delta_y = delta_y
                
        self.aligned_image = aligned_science
        return aligned_science

