import os
import numpy as np
import tensorflow as tf
import h5py
import math

def dircount(path,ex):
  list_dir = []
  list_dir = os.listdir(path)
  count = 0
  for file in list_dir:
    if file.endswith(ex): # eg: '.txt'
      count += 1
  return count


def loadimg(X, sourcepath, category, start, maxfiles):
    """
    load to matrices
    
    Arguments:
    x
    category
    start
    
    Returns: 
    results --  
    """

    spath = sourcepath + category + "/"
    k = start
    
    for name in os.listdir(spath):
        if name.endswith(".jpg"):
            
            fname = spath + name
        
            #debug
            print(str(k) + ':' + fname)
        
            image = np.array(ndimage.imread(fname, flatten=False))
            my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
            X[:, k] = np.uint8(my_image.reshape((64*64*3,)))

            k += 1
            if k>((start + maxfiles) - 1):
                break

    print("Accumulated total image files is " + str(k))
    
    return X
