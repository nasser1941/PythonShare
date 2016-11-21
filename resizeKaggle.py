import cv2
import os

dirAddress = 'train/train/YFT'
dirAddressResize = 'train/train/YFT/resize'
for filename in os.listdir(dirAddress):
        img = cv2.imread(os.path.join(dirAddress, filename))
        if img==None:
          continue
        height, width, channels = img.shape
        if (height>0 and width>0):
                res = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(dirAddressResize, filename), res)
                print 'Image: ' + filename + ' saved'