import numpy as np
import matplotlib.pyplot as plt
import cv2

# Creates an image path
img_path = '/Users/justinkeener/Desktop/Digital-Image-Class/images/blue head magician.jpg'
img = cv2.imread(img_path)

img_path2 = '/Users/justinkeener/Desktop/Digital-Image-Class/images/Spider Man.jpeg'
img2 = cv2.imread(img_path2)

# Unpacks the number of rows, columns, and channels and prints these values
rows,cols,ch = img.shape
rows2,cols2,ch2 = img2.shape

# Choosing the designated corners of coordinates
pts1 = np.float32([[164,976],[4027,1015],[164,3000],[4027,3000]])
pts3 = np.float32([[250,40],[180,40],[250,180]]) 

# Rescaling the axes of the image
pts2 = np.float32([[0,0],[1500,0],[0,1500],[1500,1500]])
pts4 = np.float32([[250,120],[180,40],[120,200]])

# 3 x 3 Transformation Matrix
M = cv2.getPerspectiveTransform(pts1,pts2)
M2 = cv2.getAffineTransform(pts3,pts4)

# Destination Image
dst = cv2.warpPerspective(img,M,(1500,1500))
dst2 = cv2.warpAffine(img2,M2,(cols2,rows2))

# Plotting the images
cv2.circle(img, (164,976), 50, (0,255,0), -1)
cv2.circle(img, (4027,1015), 50, (0,255,0) , -1)
cv2.circle(img, (164,3000), 50, (0,255,0), -1)
cv2.circle(img, (4027, 3000), 50, (0,255,0), -1)

plt.figure(figsize =(25, 25))
plt.subplot(121),plt.imshow(img),plt.title('Nonorthograpgic')
plt.subplot(122),plt.imshow(dst),plt.title('Orthographical Version')

plt.show()

plt.figure(figsize=(25,25))

cv2.circle(img2, (250,40), 5, (0,255,0), -1)
cv2.circle(img2, (180,40), 5, (0,255,0), -1)
cv2.circle(img2, (120,200), 5, (0,255,0), -1)
plt.subplot(221),plt.imshow(img2),plt.title('Spiderman')
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

plt.subplot(222),plt.imshow(dst2),plt.title('Affine Transform SpiderMan')
plt.imshow(cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB))

plt.show()