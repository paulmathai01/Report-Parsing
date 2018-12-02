import skimage.segmentation
from matplotlib import pyplot as plt
import scipy
import numpy as np
from collections import defaultdict, namedtuple
import cv2
from random import randint
from skimage import io, draw, transform
import argparse
from subprocess import call

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


class Rectangle(namedtuple('Rectangle', 'x0 y0 x1 y1')):
    def pixels(_, shape):
	    return draw.polygon([_.x0, _.x0, _.x1, _.x1], [_.y0, _.y1, _.y1, _.y0], shape)


def aabb(region):
    (x0, y0) = np.min(region, axis=0)
    (x1, y1) = np.max(region, axis=0)
    return Rectangle(x0, y0, x1, y1)

img2 = scipy.misc.imread(args["image"])
###
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,5,5)

thresh = cv2.bilateralFilter(thresh,1,75,75)
kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
thresh1 = cv2.filter2D(thresh,-1,kernel)
thresh1 = cv2.Canny(thresh1,50,150,apertureSize = 3)
#cv2.imshow("a1", thresh1)		
#cv2.waitKey(0)
"""
lines = cv2.HoughLines(thresh1,1,np.pi/180,200,1)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(thresh1,(x1,y1),(x2,y2),(0,0,0),5)
"""
#cv2.imshow("a1", thresh1)	
#cv2.waitKey(0)

###
segment_mask1 = skimage.segmentation.felzenszwalb(img2, scale=5000)
segment_mask2 = skimage.segmentation.felzenszwalb(img2, scale=5000)
segments = segment_mask2

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(segment_mask1); ax1.set_xlabel("k=100")
ax2.imshow(segment_mask2); ax2.set_xlabel("k=1000")
fig.suptitle("Felsenszwalb's efficient graph based image segmentation")
plt.tight_layout()
plt.show()

numRegions = segments.max()
rectangles = []
y = []

for regionTag in range(numRegions):
    	selectedRegion = segments == regionTag
    	regionPixelIndices = np.transpose(np.nonzero(selectedRegion))
    	(x0, y0) = np.min(regionPixelIndices, axis=0)
    	(x1, y1) = np.max(regionPixelIndices, axis=0)
    	rectangles.append(((x0, y0, x1, y1)))
    	x = aabb(regionPixelIndices)
    	y.append(x)

#print(rectangles)
a = rectangles[5]
print(len(rectangles))
"""
im1 = img2[a[0]:a[2],a[1]:a[3]]
cv2.imshow("a", im1)
cv2.waitKey(0)
"""
g =0
for i in range(len(rectangles)):
	a =rectangles[i]
	try:
		im1 = img2[a[0] - 5:a[2] + 5,a[1] -5 :a[3] + 5]
		height = a[3]-a[1]
		length = a[2]-a[0]
		if(height>length+10 or length>height+10 or length<2 or height<2):
			continue
		g = g+1
		#cv2.imwrite("/home/akash/Documents/image-classification-keras/parse_imgs/%d.png"%i,im1)
		cv2.imshow("a", im1)
		cv2.waitKey(0)

	except cv2.error:
		continue
print(g)
