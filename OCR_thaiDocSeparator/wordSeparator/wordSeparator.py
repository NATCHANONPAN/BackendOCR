import numpy as np
from .preprocess import word_preprocess
from .utils.wordSpaceDetection import strip_np, remove_sparse, sum_close_values, remove_upper_outliers

def getColHistogram(image):
  col_sum = np.sum( image, axis=0)
  return col_sum

def wordSeparator(rgb_image, p_image):
  p_img = word_preprocess(p_image)
  col_hist = getColHistogram(p_img)
  start, end  = strip_np(col_hist, value=20)
  col_hist = col_hist[start:end]

  # Find threshold
  col_hist_dense = remove_sparse(col_hist, threshold = 10)
  threshold = np.percentile(col_hist_dense, 15)

  bounding_line = sum_close_values(1*(col_hist < threshold), window_size=10)
  arr, up = remove_upper_outliers([val for id,val in bounding_line])
  space_len = max(np.percentile(arr, 90), 20)

  result = [start]
  for xc,yc in bounding_line:
    if yc >= space_len or yc in up:
      result += [xc+start+yc//4, xc+yc+start-yc//4]
  result += [end]

  imgs = []
  res=[]
  for i in range(0, len(result), 2):
      imgs.append(rgb_image[:,result[i]:result[i+1]])
      res.append((result[i],result[i+1]))

  return res
  # return imgs