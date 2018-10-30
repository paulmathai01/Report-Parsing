import cv2
import numpy as np

img = cv2.imread('abc1.png')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


########################
#Visualizations

cv2.imshow("img", closing)
cv2.waitKey(0)
cv2.imshow("img", opening)
cv2.waitKey(0)
cv2.imshow("img", img)
cv2.waitKey(0)

cv2.imshow("LSD",drawn_img )
cv2.waitKey(0)
#cv2.imwrite('houghlines5.jpg',img)

#cv2.imwrite('houghlines3.jpg',img)

########################