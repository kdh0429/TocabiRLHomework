import png
import numpy as np
import cv2

# width = 100
# height = 100
# img = np.random.uniform(0.0, 256.0 ,(width, height))
# img[49:51,49:51] = 0.0

# cv2.imwrite("noisy_terrain.png", img)

width = 100
height = 100
img = np.zeros((width, height))

cv2.imwrite("flat_terrain.png", img)