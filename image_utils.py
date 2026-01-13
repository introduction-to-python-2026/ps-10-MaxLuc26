from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def load_image(file_path):
  input_image = Image.open(file_path)
  img_array = np.array(input_image)
  return img_array

from scipy.signal import convolve2d
def edge_detection(loaded_img):
  gray_tumor = np.mean(loaded_img, axis=2)
  plt.imshow(gray_tumor, cmap='gray')

  kernelY = np.array([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]])

  kernelX = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])

  edge_x = convolve2d(gray_tumor, kernelX, mode='same', boundary='fill', fillvalue=0)
  edge_y = convolve2d(gray_tumor, kernelY, mode='same', boundary='fill', fillvalue=0)

  edge_mag = np.sqrt(edge_x**2 + edge_y**2)

  return edge_mag
