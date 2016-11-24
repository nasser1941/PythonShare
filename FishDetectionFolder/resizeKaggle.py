from __future__ import print_function
import pickle
import cv2
import os
import numpy as np
import json
from Fish import FishClass

print(os.getcwd())

dirAddress = '/home/terminale3/PycharmProjects/PythonShare/FishDetectionFolder/train/train/YFT_ORIG'
dirAddressResize = '/home/terminale3/PycharmProjects/PythonShare/FishDetectionFolder/train/train/YFT/resizew'
for filename in os.listdir(dirAddress):
        img = cv2.imread(os.path.join(dirAddress, filename))
        if img==None:
          continue
        height, width, channels = img.shape
        if (height>0 and width>0):
                res = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(dirAddressResize, filename), res)
                print ('Image: ' + filename + ' saved')