import numpy as np
import cv2

def row_preprocess(image):
  # Thinning: morphological opening
  kernel = np.ones((2, 8), np.uint8)
  image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)

  return image/255