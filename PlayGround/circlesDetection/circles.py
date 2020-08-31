import cv2
import numpy as np

planets = cv2.imread('Geometry1.png')
planets = cv2.resize(planets,(360,480))
gray_img = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray_img, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

cv2.imshow("Cirlces", img)
# center

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=30, minRadius=30, maxRadius=50)
circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    #	draw	the	outer	circle
    cv2.circle(planets, (i[0], i[1]), i[2], (0, 255, 0), 6)
    #	draw	the	center	of	the	circle
    cv2.circle(planets, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow("HoughCirlces", planets)
cv2.waitKey()
cv2.destroyAllWindows()