import cv2
import numpy as np
import matplotlib.pyplot as plt

# Creates the image path to be read into a variable
img_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/im1.jpg"
img = cv2.imread(img_path,0)

img2_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/im2.jpg"
img2 = cv2.imread(img2_path,0)

img3_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/im3.jpg"
img3 = cv2.imread(img3_path)

# Part 1

# Convert to Frequency Domain
f = np.fft.fft2(img)
f_shift = np.fft.fftshift(f)

# Convert back to Spatial
f_ishift = np.fft.ifftshift(f_shift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Plots the Original Image and the Image Converted Back
plt.figure()
plt.subplot(141),plt.imshow(img, cmap = 'gray'),plt.title('Original Image'),plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(img_back, cmap  = 'gray'),plt.title('Converted Back to Spatial Image'),plt.xticks([]), plt.yticks([])
plt.show()

# Horizontal and Vertical Edges
VE = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
HE = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)

# Vertical Edges Converted to Frequency Domain
ve = np.fft.fft2(VE)
ve_shift = np.fft.fftshift(ve)
magn_ve_shift = np.log(np.abs(ve_shift)) + 1 
ve_shift_phase = np.angle(ve_shift)

# Horizontal Edges Converted to Frequency Domain
he = np.fft.fft2(HE)
he_shift = np.fft.fftshift(he)
magn_he_shift = np.log(np.abs(he_shift)) + 1
he_shift_phase = np.angle(he_shift)

# Vertical Edges converted ack to Spatial Domain
ve_shift = np.fft.ifftshift(ve_shift)
ve_ifft = np.fft.ifft2(ve_shift)
ve_ifft = ve_ifft.real

# Horizontal Edges converted back to Spatial Domain
he_ishift = np.fft.ifftshift(he_shift)
he_ifft = np.fft.ifft2(he_ishift)
he_ifft = he_ifft.real

# Horizontal and Vertical Edges merged together to form Image 2
merged_img = cv2.add(ve_ifft,he_ifft)

# Plotting the Horizontal & Vertical Edges in Spatial Domain & Magnitude and Phase Spectrum in the Frequency Domain, and the Merged Image
plt.figure(figsize=(20,20))
plt.subplot(521),plt.imshow(VE, cmap = 'gray'),plt.title('Vertical Edges'),plt.xticks([]), plt.yticks([])
plt.subplot(522),plt.imshow(HE, cmap = 'gray'),plt.title('Horizontal Edges'),plt.xticks([]), plt.yticks([])
plt.subplot(523),plt.imshow(magn_ve_shift, cmap = 'gray'),plt.title('Vertical Magnitude Spectrum'),plt.xticks([]), plt.yticks([])
plt.subplot(524),plt.imshow(ve_shift_phase, cmap = 'gray'),plt.title('Vertical Phase'),plt.xticks([]), plt.yticks([])
plt.subplot(525),plt.imshow(magn_he_shift, cmap = 'gray'),plt.title('Horizontal Magnitude Spectrum'),plt.xticks([]), plt.yticks([])
plt.subplot(526),plt.imshow(he_shift_phase, cmap = 'gray'),plt.title('Horizontal Phase'),plt.xticks([]), plt.yticks([])
plt.subplot(527),plt.imshow(ve_ifft, cmap = 'gray'),plt.title('Converted Back Vertical Edges'),plt.xticks([]), plt.yticks([])
plt.subplot(528),plt.imshow(he_ifft, cmap = 'gray'),plt.title('Converted Back Horizontal Edges'),plt.xticks([]), plt.yticks([])
plt.subplot(529),plt.imshow(merged_img, cmap = 'gray'),plt.title('Merged Image'),plt.xticks([]), plt.yticks([])
plt.subplot(5,2,10),plt.imshow(img2, cmap = 'gray'),plt.title('Image 2'),plt.xticks([]), plt.yticks([])
plt.show()

# Part 2

# Collects data for rows and columns of the image
rows, cols = img3.shape

# Shifts the image to Frequency Domain
fft = np.fft.fft2(img)
f_shift3 = np.fft.fftshift(fft)

# Collects data to center the rows and columns 
crow,ccol = rows//2 , cols//2

# Creates a mask with size of 50x50 with center square is 1 and the remaining are all zeros
mask = np.zeros((rows,cols), np.uint8)
mask[crow-25:crow+25, ccol-25:ccol+25] = 1

# LPF: Low Pass Filter
# Applies a convolution of the Frequency Domain image and the mask to create a Low Pass Filter Image
LPF = f_shift3*mask
f_ishift3 = np.fft.ifftshift(LPF)

# Converts low pass filtered image back into Spatial Domain
img_back3 = np.fft.ifft2(f_ishift3)
img_back3 = np.abs(img_back3)

# Plot the Original Image and Low Pass Filter Image
plt.subplot(121),plt.imshow(img, cmap = 'gray'),plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray'),plt.title('Image after LPF'), plt.xticks([]), plt.yticks([])

# HPF: High Pass Filter
# Applies a window to remove all of the low frequencies to the shifted image
f_shift3[crow-15:crow+15, ccol-15:ccol+15] = 0

# Shifts image back to Spatial Domain passing through a High Pass Filter
f_ishift3 = np.fft.ifftshift(f_shift3)
hpf_img_back = np.fft.ifft2(f_ishift3)
hpf_img_back = hpf_img_back.real

# Plots Image 3 and the image after being filtered
plt.figure()
plt.subplot(121),plt.imshow(img3, cmap = 'gray'),plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(hpf_img_back, cmap = 'gray'),plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.show()

# Part 3

# Magnitude and Phase Spectrum of Image 1
magn_spectrum = np.log(np.abs(f_shift))
f_phase = np.angle(f_shift)

# Magnitude and Phase Spectrum of Image 2
magn_spectrum3 = np.log(np.abs(f_shift3))
f_phase3 = np.angle(f_shift3)

plt.subplot(221),plt.imshow(magn_spectrum, cmap = 'gray'),plt.title('Magnitude Spectrum'),plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(f_phase, cmap = 'gray'),plt.title('Phase Spectrum'),plt.xticks([]), plt.yticks([])
plt.subplot(221),plt.imshow(magn_spectrum3, cmap = 'gray'),plt.title('Magnitude Spectrum'),plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(f_phase3, cmap = 'gray'),plt.title('Phase Spectrum'),plt.xticks([]), plt.yticks([])