import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Creates the image path to be read into a variable
img_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/im1.jpg"
img = cv2.imread(img_path,0)

img2_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/im2.jpg"
img2 = cv2.imread(img2_path,0)

img3_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/im3.jpg"
img3 = cv2.imread(img3_path,0)

img4_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/im4.jpg"
img4 = cv2.imread(img4_path)

img5_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/im5.jpg"
img5 = cv2.imread(img5_path)

def swapPhase(image1,image2):
    
    # Resizes image 2
    image2 = cv2.resize(image2,(317,231))

    # Shift image 1 to Frequency Domain
    f = np.fft.fft2(image1)
    fshift = np.fft.fftshift(f)

    # Determine the magnitude and phase of Image 1 Frequency Domain
    magn_spectrum = np.log(np.abs(fshift))
    phase = np.angle(fshift)

    # Shift image 1 to Frequency Domain
    f2 = np.fft.fft2(image2)
    fshift2 = np.fft.fftshift(f2)
    
    # Determine the magnitude and phase of Image 2 Frequency Domain
    magn_spectrum2 = np.log(np.abs(fshift2))
    phase2 = np.angle(fshift2)

    # Swap the phases
    output_fimg1 = magn_spectrum*(np.exp(1j*phase2))
    output_fimg2 = magn_spectrum2*(np.exp(1j*phase))

    # Convert swapped phase images back to Spatial Domain
    ifftshift = np.fft.ifftshift(output_fimg1)
    ifft = np.fft.ifft2(ifftshift)
    ifft = ifft.real

    ifftshift2 = np.fft.ifft(output_fimg2)
    ifft2 = np.fft.ifft2(ifftshift2)
    ifft2 = ifft2.real

    # Plot images and phase spectrums
    plt.figure()
    plt.subplot(421),plt.imshow(image1,cmap= 'gray'),plt.title("Image 1"),plt.xticks([]),plt.yticks([])
    plt.subplot(422),plt.imshow(image2,cmap= 'gray'),plt.title("Image 3"),plt.xticks([]),plt.yticks([])
    plt.subplot(423),plt.imshow(magn_spectrum,cmap= 'gray'),plt.title("Image 1 Magnitude Spectrum"),plt.xticks([]),plt.yticks([])
    plt.subplot(424),plt.imshow(phase,cmap= 'gray'),plt.title("Image 1 Phase Spectrum"),plt.xticks([]),plt.yticks([])
    plt.subplot(425),plt.imshow(magn_spectrum2,cmap= 'gray'),plt.title("Image 3 Magnitude Spectrum"),plt.xticks([]),plt.yticks([])
    plt.subplot(426),plt.imshow(phase2,cmap= 'gray'),plt.title("Image 3 Phase Spectrum"),plt.xticks([]),plt.yticks([])
    plt.subplot(427),plt.imshow(ifft,cmap= 'gray'),plt.title("Magnitude Image 1 with Image 3 Phase"),plt.xticks([]),plt.yticks([])
    plt.subplot(428),plt.imshow(ifft2,cmap= 'gray'),plt.title("Magnitude Image 3 with Image 1 Phase"),plt.xticks([]),plt.yticks([])
    plt.show()
swapPhase(img,img3)