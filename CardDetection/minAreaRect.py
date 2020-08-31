import numpy as np
import argparse
import cv2
import sys
from matplotlib import image as PLT

scale_percent=100
kernelopen  = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernelclose = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))

#kernel2 = np.ones((7,7),np.float32)/49

#cv2.MORPH_ELLIPSE
#cv2.MORPH_CROSS
#cv2.MORPH_RECT
  



image = cv2.imread('testImage/image1.png')
image =image[:][range(int(image.shape[0]/2))][:]
image_before=image
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

image =cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernelclose)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernelopen)

coords = np.column_stack(np.where(opening > 0))
point = cv2.minAreaRect(coords)




box=np.int0(cv2.boxPoints(point))
print(type(box))
print(box)
keep =[] 

for i in box:
	keep.append([int(i[1]),int(i[0])])

keep = np.array(keep)
print(type(keep))
print(keep)


cv2.drawContours(image,[keep],0,(0,0,255),1)
print(coords)


for i in coords:
	x=int(i[1])
	y=int(i[0])
	cv2.circle(image,(x,y),3,(0,255,0),1)




angle = point[-1]
if angle < -45:
	angle = -(90 + angle)
else:
	angle = -angle
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
print("[INFO] angle: {:.3f}".format(angle))

center =(w//2,h//2)
M =cv2.getRotationMatrix2D(center,angle,1.0)
img_rotated = cv2.warpAffine(image_before,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)


ShowImageRotated=((image+rotated))
cv2.imshow("thresh",thresh)
cv2.imshow("Input", image)
cv2.imshow("Rotated", rotated)
cv2.imshow("ShowImageRotated",ShowImageRotated)
cv2.imshow("closing",closing)
cv2.imshow("opening",opening)
cv2.imshow("output",img_rotated)


PLT.imsave('output/image/output.png', img_rotated)
PLT.imsave('output/image/thresh.png', thresh)
PLT.imsave('output/image/Rotated.png', rotated)
PLT.imsave('output/image/ShowImageRotated.png', ShowImageRotated)
PLT.imsave('output/image/closing.png', closing)
PLT.imsave('output/image/opening.png', opening)

print(thresh.shape)
print(cv2.minAreaRect(coords))


cv2.waitKey(0)