from PIL import Image, ImageEnhance
from PIL import ImageFilter
import pandas
import matplotlib.pyplot as plt
import numpy as numpy
from scipy.stats import norm
from scipy.stats import skew
import scipy as sp
from scipy import stats
import pylab as pl
import cv2

def grades_sum(my_list):
    total = 0
    for grade in my_list:
        total += grade
    return total

def grades_average(my_list):
    sum_of_grades = grades_sum(my_list)
    average = sum_of_grades / len(my_list)
    return average

def grades_variance(my_list, average):
    variance = 0
    for i in my_list:
        variance += (average - my_list[i]) ** 2
    return variance / len(my_list)


imgGray = Image.open('images/tiger.jpg').convert('L')
pixels = imgGray.load() # create the pixel map

fpixels = []

for i in range(0,13):    # for every pixel:
    for j in range(0,13):
        fpixels.append(pixels[i,j])


spixels = []
for i in range(0,13):    # for every pixel:
    for j in range(0,13):
        spixels.append(pixels[i+200,j+200])


print ('first sample:', fpixels)
print ('second sample:', spixels)

averagef = grades_average(fpixels)
averages = grades_average(spixels)

skewf =  skew(fpixels)
skews =  skew(spixels)


nf, min_maxf, meanf, varf, skewf, kurtf = stats.describe(fpixels)
ns, min_maxs, means, vars, skews, kurts = stats.describe(spixels)
print('meanf:', meanf)
print('means:', means)
print('varf:', varf)
print('vars:', vars)
print('skewf:', skewf)
print('skews:', skews)
print('kurtf:', kurtf)
print('kurts:', kurts)
#variancef = grades_variance(fpixels, averagef)
#variances = grades_variance(spixels, averages)

ax = pl.subplot(111)
#ax.bar(2, meanf, width=1)
#ax.bar(4, means, width=1)
ax.bar(8, varf, width=1)
ax.bar(10, vars, width=1)
#ax.bar(14, skewf, width=1)
#ax.bar(16, skews, width=1)
#ax.bar(20, kurtf, width=1)
#ax.bar(22, kurts, width=1)

pl.show()