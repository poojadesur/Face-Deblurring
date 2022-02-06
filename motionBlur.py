import cv2
import numpy as np
from scipy.signal import convolve2d
import os
from imageio import imread
filename = os.path.join(os.path.dirname(__file__), 'input.png')
img = imread(filename, as_gray=True)


kernel_size = 30
kernel_v = np.zeros((kernel_size, kernel_size))

kernel_h = np.copy(kernel_v)

kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)

kernel_v /= kernel_size
kernel_h /= kernel_size

vertical_mb = cv2.filter2D(img, -1, kernel_v)

horizonal_mb = cv2.filter2D(img, -1, kernel_h)

y = convolve2d(img, kernel_h, mode='same', boundary='wrap')

cv2.imwrite('blur_vertical.jpg', vertical_mb)
cv2.imwrite('blur_horizontal.jpg', y)
