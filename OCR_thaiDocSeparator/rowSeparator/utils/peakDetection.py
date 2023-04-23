import numpy as np
from peakdetect import peakdetect

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def peakDetection(y, p_threshold=-1, v_threshold=-1):
  [peaks_raw, valleys_raw] = peakdetect(y, lookahead=50)
  if peaks_raw == []: return 
  peaks = []
  valleys=[]

  for peak in peaks_raw:
    if p_threshold == -1 or peak[1] > p_threshold:
      peaks.append(peak[0])

  for valley in valleys_raw:
    if v_threshold == -1 or valley[1] < v_threshold:
      valleys.append(valley[0])
  
  return peaks, valleys