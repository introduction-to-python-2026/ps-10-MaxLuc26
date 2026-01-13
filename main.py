!wget https://raw.githubusercontent.com/yotam-biu/ps12/main/image_utils.py -O /content/image_utils.py
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit

from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import numpy as np
import matplotlib.pyplot as plt

def load_image(file_path):
  input_image = Image.open(file_path)
  img_array = np.array(input_image)
  return img_array

file_path = "/content/the fire rises photo.jfif" #testing the function
loaded_img = load_image(file_path)

print(loaded_img.shape) #removing red values i think
ready_img = np.mean(loaded_img, axis=2)
plt.imshow(ready_img)
plt.show()

plt.imshow(loaded_img) #regular loaded img
plt.show()

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

plt.imshow(edge_detection(loaded_img), cmap='Spectral_r')

loaded_img = load_image(file_path)
print(loaded_img.shape)

import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image

new_img = load_image("/content/brain_tumor.jpg")

clean_image = median(new_img, ball(3))
step_before_binary_image = edge_detection(clean_image)
binary_image = step_before_binary_image > 5
plt.imshow(binary_image, cmap='gray')
plt.show()

edge_image_pil = Image.fromarray(binary_image)
edge_image_pil.save('my_edges.png')
