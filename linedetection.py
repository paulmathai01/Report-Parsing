import cv2
import numpy as np

img = cv2.imread('abc1.png')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
x = gray.copy()
kernel = np.ones((3,3),np.uint8)
dilation = cv2.dilate(img,kernel,iterations = 5)
edges = cv2.Canny(gray,50,150,apertureSize = 3)


#img = cv2.imread("test.png",0)


lsd = cv2.createLineSegmentDetector(0)
lines = lsd.detect(edges)[0] #Position 0 of the returned tuple are the detected lines
drawn_img = lsd.drawSegments(img,lines)

img_hsv=cv2.cvtColor(drawn_img, cv2.COLOR_BGR2HSV)

lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

mask = mask0+mask1

output_img = img.copy()
output_img[np.where(mask==0)] = 0

kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(output_img, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
kernel1 = np.ones((40,40),np.uint8)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel1)

m=cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
x[np.where(m>0)] = 255
x[np.where(x>200)] = 255
#x[np.where(x<200)] = 0
########################
#Visualizations

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.imshow("img", mask)
cv2.waitKey(0)
cv2.imshow("img", closing)
cv2.waitKey(0)

cv2.imshow("LSD",x)
cv2.waitKey(0)
#cv2.imwrite('houghlines5.jpg',img)

#cv2.imwrite('houghlines3.jpg',img)

########################