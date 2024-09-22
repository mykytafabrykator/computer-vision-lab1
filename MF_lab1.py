import os

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, img_as_float
from scipy.signal import convolve2d
from scipy.ndimage import median_filter


if not os.path.exists("result_images"):
    os.makedirs("result_images")


def save_image(image, filename):
    plt.imsave(f"result_images/{filename}.png", image, cmap="gray")


def negative(image):
    image = img_as_float(image)
    neg_image = 1 - image
    save_image(neg_image, "negative")
    return neg_image


def log_transform(image, c=30):
    image = img_as_float(image) * 255.0
    epsilon = 1e-5

    log_image = c * np.log(1 + image + epsilon)
    log_image = (log_image / np.max(log_image)) * 255.0

    save_image(log_image, "logarithmic_transform")
    return log_image


def power_transform(image, gamma=0.3, c=50):
    image = img_as_float(image)
    power_image = c * np.power(image, gamma)
    power_image = (power_image - np.min(power_image)) / (
        np.max(power_image) - np.min(power_image)
    )

    save_image(power_image, "power_transform")
    return power_image


def contrast_stretch(image, m=0.5, E=5):
    image = img_as_float(image)
    stretched_image = 1 / (1 + np.power(m / (image + 1e-5), E))
    stretched_image = (stretched_image - np.min(stretched_image)) / (
        np.max(stretched_image) - np.min(stretched_image)
    )

    save_image(stretched_image, "contrast_stretching")
    return stretched_image


def histogram_equalization(image):
    image = img_as_float(image)
    hist_eq_image = exposure.equalize_hist(image)
    hist_eq_image = (hist_eq_image - np.min(hist_eq_image)) / (
        np.max(hist_eq_image) - np.min(hist_eq_image)
    )

    save_image(hist_eq_image, "histogram_equalization")
    return hist_eq_image


def average_filter(image, size=3):
    kernel = np.ones((size, size)) / (size * size)
    smooth_image = convolve2d(image, kernel, mode="same")
    save_image(smooth_image, "averaging_filter")
    return smooth_image


def laplacian_sharpening(image):
    image = img_as_float(image)
    laplace_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    laplace_image = convolve2d(image, laplace_kernel, mode="same")
    sharpened_image = image - laplace_image
    sharpened_image = np.clip(sharpened_image, 0, 1)

    save_image(sharpened_image, "laplacian_sharpening")
    return sharpened_image


def gradient_processing(image):
    image = img_as_float(image)
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    grad_x = convolve2d(image, sobel_x, mode="same")
    grad_y = convolve2d(image, sobel_y, mode="same")
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_magnitude = np.clip(grad_magnitude, 0, 1)

    save_image(grad_magnitude, "gradient_processing")
    return gradient_processing


def median_filtering(image, size=3):
    median_image = median_filter(image, size=size)
    save_image(median_image, "median_filtering")
    return median_image


if __name__ == "__main__":
    pic1 = io.imread("images/pic1.jpg", as_gray=True)
    pic2 = io.imread("images/pic2.jpg", as_gray=True)
    pic3 = io.imread("images/pic3.jpg", as_gray=True)
    pic4 = io.imread("images/pic4.jpg", as_gray=True)
    pic5 = io.imread("images/pic5.jpg", as_gray=True)
    pic6 = io.imread("images/pic6.jpg", as_gray=True)
    pic7 = io.imread("images/pic7.jpg", as_gray=True)
    pic8 = io.imread("images/pic8.jpg", as_gray=True)
    pic9 = io.imread("images/pic9.jpg", as_gray=True)

    negative(pic1)
    log_transform(pic2)
    power_transform(pic3)
    contrast_stretch(pic4)
    histogram_equalization(pic5)
    average_filter(pic6)
    laplacian_sharpening(pic7)
    gradient_processing(pic8)
    median_filtering(pic9)
