import numpy as np

def strip_np(arr, value=20): # strip out lower than parameter value
  start_index = np.where(arr > value)[0][0]
  end_index = np.where(arr > value)[0][-1]
  return start_index, end_index

def remove_sparse(arr, threshold):
  return arr[arr >= threshold]

def sum_close_values(arr, window_size):
  np_arr = np.array(arr)
  result = []
  skip=0
  for i in np.nonzero(np_arr)[0]:
    if skip>0:
      skip-=1
      continue

    val = np_arr[i]
    start = i
    count=1
    while(count<window_size):
      if start+count < len(arr) and arr[start+count] !=0:
        val += arr[start+count]
        start += count
        count=0
        skip+=1
      count+=1
    result.append((i,val))
  if result==[]:
    return [(0,len(arr)-1)]
  return result

def remove_upper_outliers(data):
  data = np.unique(data)
  q1, q3 = np.percentile(data, [25, 75])
  iqr = q3 - q1
  upper_bound = q3 + 1.5*iqr
  return data[data <= upper_bound], data[(data > upper_bound)]