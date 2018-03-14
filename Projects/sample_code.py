import cv2
import numpy as np
import matplotlib.pyplot as plt

# Creates the image path to be read into a variable
img_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/im1.jpg"
img = cv2.imread(img_path,0)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)

f = np.fft.fft2(img)
f_shift = np.fft.fftshift(f)
magn_spectrum = 20*np.log(np.abs(f_shift))

f_ishift = np.fft.ifftshift(f_shift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
plt.subplot(221),plt.imshow(img, cmap = 'gray'),plt.title('Original Image')
plt.subplot(222),plt.imshow(magn_spectrum, cmap = 'gray'),plt.title('Magnitude Spectrum')
plt.subplot(223),plt.imshow(img_back, cmap = 'gray'),plt.title('Converted back to Original Image')
plt.show()