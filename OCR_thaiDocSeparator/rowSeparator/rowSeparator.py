import numpy as np
from .preprocess import row_preprocess
from .utils.peakDetection import smooth, peakDetection

def getRowHistogram(image):
  row_sum = np.sum( image, axis=1)

  # Clean head and tail part
  # row_sum[:1400] = 0
  # row_sum[-50:] = 0

  return row_sum

PADDING = 40
BOUND = 100

def getBoundingLines(peaks, valleys):
  valleys = np.array([0] + valleys + [peaks[-1]+BOUND])
  peaks = np.array(peaks)
  result = []
  for peak in peaks:
    for id in range(len(valleys)):
      if valleys[id] - peak > 0:
        lb = min(valleys[id], peak+BOUND)
        ub = max(valleys[id-1], peak-BOUND)
        break
    ub = max(0, ub - PADDING)
    lb = lb + PADDING
    result.append((ub,lb))
  return result

def rowSeparator(p_image):
    p_image = row_preprocess(p_image)
    row_hist = getRowHistogram(p_image)
    row_hist = smooth(row_hist, box_pts=100)
    peaks, valleys = peakDetection(row_hist, p_threshold=16)

    # Only one row
    if peaks==[] or valleys==[]:
      return [(0,p_image.shape[0])]
    
    return [ (y_upper, y_lower) for (y_upper,y_lower) in getBoundingLines(peaks,valleys)]