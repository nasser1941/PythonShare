from scipy import ndimage, misc
import os

def rotateImage2(image, angle):
    return ndimage.rotate(image, angle)

dirAddress = 'train/train/SHARK'
for filename in os.listdir(dirAddress):
        image = misc.imread(os.path.join(dirAddress, filename))
        image90 = rotateImage2(image, 90)
        misc.imsave(os.path.join(dirAddress, filename)+ '90.jpg',image90)
        image180 = rotateImage2(image90, 90)
        misc.imsave(os.path.join(dirAddress, filename)+ '180.jpg',image180)
        image270 = rotateImage2(image180, 90)
        misc.imsave(os.path.join(dirAddress, filename)+ '270.jpg',image270)
        print 'Image: ' + filename + ' Rotated and saved'




