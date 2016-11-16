import cv2
import os

dirAddress = 'train/train/SHARK'
dirAddressResize = 'train/train/SHARK/resize'
for filename in os.listdir(dirAddress):
        img = cv2.imread(os.path.join(dirAddress, filename))
        res = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(dirAddressResize, filename), res)
        print 'Image: ' + filename + ' resized and saved'