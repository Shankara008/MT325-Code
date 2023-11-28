import numpy as np
import cv2 as cv

blank = np.zeros((500,500,3),dtype = 'uint8') #uint8 is a datatype for images... here 3 in number of color channels...
blank[200:300,300:400] = 0,255,0 #color is B:G:R
#cv.imshow('green',blank)

#rectangle
cv.rectangle(blank,(0,0),(250,250),(0,255,0),thickness=3) #use -1 to fill
#cv.imshow('rectangle',blank)

#Circle
cv.circle(blank,(250,250),50,(255,0,0),thickness= 3) #to fill use -1 for thickness...
#cv.imshow('Circle',blank)

cv.line(blank,(0,0),(250,250),(255,255,255),thickness= 3)
#cv.imshow('Line',blank)

#write text
cv.putText(blank,'Fabric Shrinkage',(100,30),cv.FONT_HERSHEY_TRIPLEX,1.0,(255,255,255),thickness=2)
#cv.imshow('text',blank)

def reSize(frame,scale = 0.25):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)


#converting an image to grayscale...
image = cv.imread('Fabric/cat.jpg')
image = reSize(image)
#cv.imshow('cat',image)

image2 = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
#cv.imshow('blandwh',image2)

#reduce noise by using blur...
blurred_image = cv.GaussianBlur(image,(3,3),cv.BORDER_DEFAULT)
#cv.imshow('blurred_image',blurred_image)

#Edge cascade
canny = cv.Canny(image,125,175) # we can use the blur with this to get lesser details...
#cv.imshow('canny',canny)

#dialating the image
dialated = cv.dilate(canny,(7,7),iterations=3)
#cv.imshow('dialated',dialated)

#eroding the image #reversing the dialation
eroded = cv.erode(dialated,(7,7),iterations=3)
#cv.imshow('eroded',eroded)

#resizing the image
image3 = cv.resize(canny,(300,300),interpolation=cv.INTER_AREA) #if we are enlarging the image use cv.INTER_LINEAR or cv_INTER_CUBIC
#cv.imshow('resized',image3)

#cropping
cropped = image[200:600,200:600]
#cv.imshow('cropped',cropped)

#image transformation...

#translation (moving the image)
def translate(img,x,y):
    transmat = np.float32([[1,0,x],[0,1,y]])
    dimensions1 = (img.shape[1],img.shape[0]) #image.shape(0) = width, image.shape(1) = height
    # -x = left, -y = up
    # -x = right, y = down
    return cv.warpAffine(img,transmat,dimensions1)

translated = translate(image,100,100)
#cv.imshow('translated',translated)

#Rotation
#flipping
flipped = cv.flip(image,1)#1 is horizontal, -1 is vertical
#cv.imshow('flipped',flipped) 


#contours
blur = cv.GaussianBlur(image,(5,5),0)
cv.imshow('canny1',canny)




cv.waitKey(0)
