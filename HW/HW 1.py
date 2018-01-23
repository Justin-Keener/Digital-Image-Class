import os
import matplotlib.pyplot as plt
import cv2

# Declares the images
im1_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/ngc6357_4k.jpg"
im2_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/ngc6357_4k.png"

im1 = cv2.imread(im1_path)
im2 = cv2.imread(im2_path)

# Displays two subplot images into one figure
plt.figure(1)
plt.subplot(121)
plt.title('Chandra Galaxy.jpg')
plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))

plt.subplot(122)
plt.title('Chandra Galaxy.png')
plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
plt.show()

# Prints the infomartion to compare the file sizes
print('Size of the jpeg file in bytes: ', os.stat(im1_path).st_size)
print('Size of the png file in bytes: ', os.stat(im2_path).st_size)