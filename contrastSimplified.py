from PIL import Image, ImageEnhance
from PIL import ImageFilter
import pandas
import matplotlib.pyplot as plt
import numpy as numpy
from scipy.stats import norm
from scipy import ndimage

image2 = Image.open('images/source/image480.jpg')
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


imgGray = Image.open('images/source/image480.jpg').convert('L')

pixels = imgGray.load() # create the pixel map
histo = imgGray.histogram()
print histo
mu, std = norm.fit(histo)
plt.hist(histo, bins=25, normed=False, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = numpy.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)
plt.show()

import operator

def equalize(h):

    lut = []

    for b in range(0, len(h), 256):

        # step size
        step = reduce(operator.add, h[b:b+256]) / 255

        # create equalization lookup table
        n = 0
        for i in range(256):
            lut.append(n / step)
            n = n + h[i+b]

    return lut

# calculate lookup table
    lut = equalize(histo)

    # map image through lookup table
    imgGrayStreched = imgGray.point(lut)
    imgGrayStreched.show()

    histo2 = imgGrayStreched.histogram()
    print histo2
    mu2, std2 = norm.fit(histo2)
    plt.hist(histo2, bins=25, normed=False, color='g')

    # Plot the PDF.
    xmin2, xmax2 = plt.xlim()
    x2 = numpy.linspace(xmin2, xmax2, 100)
    p2 = norm.pdf(x2, mu2, std2)
    plt.plot(x2, p2, 'k', linewidth=2)
    title2 = "Fit results: mu = %.2f,  std = %.2f" % (mu2, std2)
    plt.title(title2)
    plt.show()

   # imgGray.save("out.ppm")

for i in range(imgGray.size[0]):    # for every pixel:
    for j in range(imgGray.size[1]):
        if int(pixels[i,j]) < 135:
            pixels[i, j] = (int(pixels[i,j]) - int(50))
        elif int(pixels[i,j]) >115:
            pixels[i, j] = (int(pixels[i, j]) + int(50))

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

        pixels[i, j] = ((int(pixels[i, j])) + (int(pixels[i,maxJ])) + (int(pixels[maxI,j])) + (int(pixels[minI,j])) + (int(pixels[i,minJ])))/(int (5))

imgGray.show()


imgGray2 = Image.open('images/source/image480.jpg').convert('L')
f = numpy.fft.fft2(imgGray2)                    #do the fourier transform
fshift1 = numpy.fft.fftshift(f)                 #shift the zero to the center
#f_ishift = numpy.fft.ifftshift(fshift1)        #inverse shift
img_back = numpy.fft.ifft2(fshift1)             #inverse fourier transform
img_back = numpy.abs(img_back)
imageGrayFourierVersion = Image.fromarray(img_back)
#imageGrayFourierVersion.show()

def plot(data, title):
    plot.i += 1
    plt.subplot(2,2,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
plot.i = 0

data = numpy.array(imgGray2, dtype=float)
im = Image.fromarray(data)
plot(data, 'Original')

kernel = numpy.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

highpass_3x3 = ndimage.convolve(data, kernel)
plot(highpass_3x3, 'Simple 3x3 Highpass')
print highpass_3x3
