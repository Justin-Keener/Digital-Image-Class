import os
import matplotlib.pyplot as plt
import cv2

# Declares the images
im_1 = cv2.imread("/Users/justinkeener/Desktop/Digital Image Pattern/HW/HW 1 /ngc6357_4k.jpg")
im_2 = cv2.imread("/Users/justinkeener/Desktop/Digital Image Pattern/HW/HW 1 /ngc6357_4k.png")

# Displays two subplot images into one figure
plt.figure(1)
plt.subplot(121)
plt.title('Chandra Galaxy.jpg')
plt.imshow(cv2.cvtColor(im_1, cv2.COLOR_BGR2RGB))

plt.subplot(122)
plt.title('Chandra Galaxy.png')
plt.imshow(cv2.cvtColor(im_2, cv2.COLOR_BGR2RGB))
plt.show()

# Prints the infomartion to compare the file sizes
print('Size of the jpeg file in bytes: ', os.stat("/Users/justinkeener/Desktop/Digital Image Pattern/HW/HW 1 /ngc6357_4k.jpg").st_size)
print('Size of the png file in bytes: ', os.stat("/Users/justinkeener/Desktop/Digital Image Pattern/HW/HW 1 /ngc6357_4k.png").st_size)