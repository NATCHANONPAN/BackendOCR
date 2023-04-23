import numpy as np
import cv2

def word_preprocess(image):
  # Thinning: morphological
  kernel = np.ones((3, 3), np.uint8)
  image = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel, iterations=1)

  # Extract Core zone
  h,w = image.shape
  image = image[h//4:3*h//4, :]
  return image/255