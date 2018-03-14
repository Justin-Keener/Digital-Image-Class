import cv2
import numpy as np
import matplotlib.pyplot as plt

# Creates the image path to be read into a variable
img_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/im1.jpg"
img = cv2.imread(img_path,0)

# Part 1
# Convert to Frequency Domain
f = np.fft.fft2(img)
f_shift = np.fft.fftshift(f)
magn_spectrum = 20*np.log(np.abs(f_shift))

# Convert back to Spatial
f_ishift = np.fft.ifftshift(f_shift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.figure()
plt.subplot(131),plt.imshow(img, cmap = 'gray'),plt.title('Original Image')
plt.subplot(132),plt.imshow(magn_spectrum, cmap = 'gray'),plt.title('Magnitude Spectrum')
plt.subplot(133),plt.imshow(img_back, cmap  = 'gray'),plt.title('Converted Back to Spatial Image')
plt.show()

# Horizontal and Vertical Edges
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)

# Frequency Domain Horizontal and Vertical Edges 
f_sobelx = np.fft.fft2(sobelx)
f_sobelx_shift = np.fft.fftshift(f_sobelx)
magn_spect_x = np.log(np.abs(f_sobelx_shift))

f_sobely = np.fft.fft2(sobely)
f_sobely_shift = np.fft.fftshift(f_sobely)
magn_spect_y = np.log(np.abs(f_sobely_shift))

# Convert sobel x back to Frequency Domain
f_ishift_sobelx = np.fft.ifftshift(f_sobelx_shift)
sobelx_back = np.fft.ifft2(f_ishift_sobelx)
sobelx_back = np.abs(sobelx_back)

# Convert sobel y to Frequency Domain
f_ishift_sobely = np.fft.ifftshift(f_sobely_shift)
sobely_back = np.fft.ifft2(f_ishift_sobely)
sobely_back = np.abs(sobely_back)

# Merged image of the converted back to spatial domain horizontal and vertical edges
merged_img = cv2.add(sobelx_back,sobely_back)

# Plotting Sobel, Converted Back Sobel, and Merged Image
plt.figure(figsize=(20,20))
plt.subplot(321),plt.imshow(sobelx, cmap = 'gray'),plt.title('Sobel X')
plt.subplot(322),plt.imshow(sobely, cmap = 'gray'),plt.title('Sobel Y')
plt.subplot(323),plt.imshow(sobelx_back, cmap = 'gray'),plt.title('Converted Back Sobel X')
plt.subplot(324),plt.imshow(sobely_back, cmap = 'gray'),plt.title('Converted Back Sobel Y')
plt.subplot(325),plt.imshow(merged_img, cmap = 'gray'),plt.title('Merged Image')
plt.show()