from PIL import Image, ImageEnhance
from PIL import ImageFilter
import pandas
import matplotlib.pyplot as plt
import numpy as numpy

image2 = Image.open('images/simple.png')
contrast = ImageEnhance.Contrast(image2)
#image2.show()
#contrast.enhance(2).show()

sharpn = ImageEnhance.Sharpness(image2)
#sharpn.enhance(2).show()

brightn = ImageEnhance.Brightness(image2)
#brightn.enhance(2).show()

im_sharp = image2.filter(ImageFilter.SHARPEN)
im_sharp.save('images/image_sharpened.jpg', 'JPEG')

im_smooth = image2.filter(ImageFilter.SMOOTH)
im_smooth.save('images/image_smoothed.jpg', 'JPEG')

im_edgeEnhance = image2.filter(ImageFilter.EDGE_ENHANCE_MORE)
im_edgeEnhance.save('images/image_edgeEnhance.jpg', 'JPEG')

im_edgefind = image2.filter(ImageFilter.FIND_EDGES)
im_edgefind.save('images/image_edgeFind.jpg', 'JPEG')


imgGray = Image.open('images/image480.jpg').convert('LA')

pixels = imgGray.load() # create the pixel map

for i in range(imgGray.size[0]):    # for every pixel:
    for j in range(imgGray.size[1]):
        if int(pixels[i,j][0]) < 125:
            pixels[i, j] = ((int(pixels[i,j][0]) - int(50)),  pixels[i, j][1])
        elif int(pixels[i,j][0]) >125:
            pixels[i, j] = ((int(pixels[i, j][0]) + int(50)),  pixels[i, j][1])

imgGray.show()

for i in range(imgGray.size[0]):    # for every pixel:
    for j in range(imgGray.size[1]):

        if i+1 > imgGray.size[0]-1:
            maxI = imgGray.size[0]-1
        else:
            maxI = i+1

        if j+1 > imgGray.size[1]-1:
            maxJ = imgGray.size[1]-1
        else:
            maxJ = j+1

        if i-1 < 0:
            minI = 0
        else:
            minI = i-1

        if j-1 < 0:
            minJ = 0
        else:
            minJ = j-1

        pixels[i, j] = (((int(pixels[i, j][0])) + (int(pixels[i,maxJ][0])) + (int(pixels[maxI,j][0])) + (int(pixels[minI,j][0])) + (int(pixels[i,minJ][0])))/(int (5)), pixels[i, j][1])

imgGray.show()


imgGray2 = Image.open('images/image480.jpg').convert('L')
f = numpy.fft.fft2(imgGray2)                    #do the fourier transform
fshift1 = numpy.fft.fftshift(f)                 #shift the zero to the center
#f_ishift = numpy.fft.ifftshift(fshift1)        #inverse shift
img_back = numpy.fft.ifft2(fshift1)             #inverse fourier transform
img_back = numpy.abs(img_back)
imageGrayFourierVersion = Image.fromarray(img_back)
imageGrayFourierVersion.show()

