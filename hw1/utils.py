import cv2
import numpy as np

# You can use functions in this file freely.

def safe_subtract(x1, x2):
    _ret =  cv2.subtract(x1, x2)
    return _ret

def safe_add(x1, x2):
    _ret = cv2.add(x1, x2)
    return _ret

def down_sampling(x1):
    _ret = cv2.pyrDown(x1)
    return _ret

def up_sampling(x1):
    _ret = cv2.pyrUp(x1)
    return _ret


