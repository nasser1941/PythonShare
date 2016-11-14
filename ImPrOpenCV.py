import cv2
import numpy as np
from scipy import ndimage
from pylab import array, plot, show, axis, arange, figure, uint8
from PIL import Image, ImageEnhance
from PIL import ImageFilter
from matplotlib import pyplot as plt

#Read Image in OpenCV
img = cv2.imread('images/source/scan001.jpg',0)

#rotation of an image by the specified angle
def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
  return result

#rotated = rotateImage(img, 1)


def rotateImage2(image, angle):
    return ndimage.rotate(image, angle)

#scale
height, width = img.shape[:2]
res = cv2.resize(img,(width/2, height/2), interpolation = cv2.INTER_CUBIC)
cv2.imwrite('images/scaleedby2.png',res)

#Morphology
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(img,kernel,iterations = 1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)



#Write final result
cv2.imwrite('images/binaryderosion.png',erosion)
cv2.imwrite('images/binaryopening.png',opening)
cv2.imwrite('images/binaryclosing.png',closing)
cv2.imwrite('images/binarydilation.png',dilation)

image2 = Image.open('images/binaryopening.png')
contrastOpening = ImageEnhance.Contrast(image2)
contrastOpening.enhance(2).save('images/contrastOpening.png', 'PNG')


def compute_skew(image):
    #image = cv2.bitwise_not(image)
    height, width = image.shape

    edges = cv2.Canny(image, 150, 200, 3, 5, True)
    cv2.imwrite('images/edges.png', edges)
    lines = cv2.HoughLinesP(edges, 1, 3.141592653589793/180, 100, minLineLength=width / 1.0, maxLineGap=10)
    angle = 0.0
    nlines = lines.size
    for x1, y1, x2, y2 in lines[0]:
        angle += np.arctan2(x2 - x1, y2 - y1)

    print angle / nlines
    return angle / nlines

img = cv2.imread('images/contrastOpening.png',0)
deskewed_image = rotateImage2(img, compute_skew(img))
cv2.imwrite('images/deskewed_image.png',deskewed_image)