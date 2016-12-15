#these values set how sensitive the bounding box detection is
threshold = 150     #the average of the darkest values must be _below_ this to count (0 is darkest, 255 is lightest)
obviousness = 20    #how many of the darkest pixels to include (1 would mean a single dark pixel triggers it)

from PIL import Image
import numpy as np
from deskew import Deskew
import cv2

def find_line(vals):
    #implement edge detection once, use many times
    for i,tmp in enumerate(vals):
        tmp.sort()
        average = float(sum(tmp[:obviousness]))/len(tmp[:obviousness])
        if average <= threshold:
            return i
    return i    #i is left over from failed threshold finding, it is the bounds

def getbox(img):
    #get the bounding box of the interesting part of a PIL image object
    #this is done by getting the darekest of the R, G or B value of each pixel
    #and finding were the edge gest dark/colored enough
    #returns a tuple of (left,upper,right,lower)

    width, height = img.size    #for making a 2d array
    retval = [0,0,width,height] #values will be disposed of, but this is a black image's box

    pixels = list(img.getdata())
    vals = []                   #store the value of the darkest color
    for pixel in pixels:
        vals.append(min(pixel)) #the darkest of the R,G or B values

    #make 2d array
    vals = np.array([vals[i * width:(i + 1) * width] for i in xrange(height)])

    #start with upper bounds
    forupper = vals.copy()
    retval[1] = find_line(forupper)

    #next, do lower bounds
    forlower = vals.copy()
    forlower = np.flipud(forlower)
    retval[3] = height - find_line(forlower)

    #left edge, same as before but roatate the data so left edge is top edge
    forleft = vals.copy()
    forleft = np.swapaxes(forleft,0,1)
    retval[0] = find_line(forleft)

    #and right edge is bottom edge of rotated array
    forright = vals.copy()
    forright = np.swapaxes(forright,0,1)
    forright = np.flipud(forright)
    retval[2] = width - find_line(forright)

    if retval[0] >= retval[2] or retval[1] >= retval[3]:
        print "error, bounding box is not legit"
        return None
    return tuple(retval)

def detectFace(docImage):
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(docImage, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # if we could not find any face we try to rotate the image 180 degree and apply the face detection again
    height, width = gray.shape[:2]
    center = (width / 2, height / 2)
    if (len(faces) == 0):
        # rotate the image by 180 degrees
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        rotated = cv2.warpAffine(gray, M, (width, height))
        faces = faceCascade.detectMultiScale(
            rotated,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # if a face find in this case we consider this face and image and continue our job
        if (len(faces) != 0):
            M = cv2.getRotationMatrix2D(center, 180, 1.0)
            docImage = cv2.warpAffine(docImage, M, (width, height))

    print("Found {0} faces!".format(len(faces)))
    return docImage, faces
#################################################################
#################################################################
inputAddress = 'patent3.jpg'
outputAddress = 'patent3Ro.jpg'

des = Deskew(inputAddress, False, outputAddress, 0)
des.run()
image = Image.open(outputAddress)

box = getbox(image)
result = image.crop(box)
result.save('sampleCut.jpg')



# Read the image
image = cv2.imread('sampleCut.jpg')
image, faces = detectFace(image)

# we keep only the biggest face. because sometime some noise fool the algorithm
faceArea = 0
faceX = 0
faceY = 0
faceW = 0
faceH = 0

for (x, y, w, h) in faces:
    print("w*h", w*h)
    if (w*h>faceArea):
        faceArea = w*h
        faceX = x
        faceY = y
        faceW = w
        faceH = h

# Draw a rectangle around the bigest faces
cv2.rectangle(image, (faceX, faceY), (faceX+faceW, faceY+faceH), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)

# we check again at the end to see if the image is upside down
height, width = image.shape[:2]
center = (width / 2, height / 2)
print("hight: ", height, "width: ", width, "FaceY: ", faceY)
if(faceY>(height/2) or len(faces)==0):
    # rotate the image by 180 degrees
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(image, M, (width, height))
    cv2.imshow("rotated", rotated)
    cv2.waitKey(0)

documentType = ''
if(faceX>(width / 2)):
    documentType = 'cartaDiIdentita'
else:
    documentType = 'patente'

print ("Document type is: ", documentType)
