import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/im6.jpg"
img = cv2.imread(img_path,0)

img6_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/im6.jpg"
img6 = cv2.imread(img6_path,0)

img7_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/im7.jpg"
img7 = cv2.imread(img7_path,0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
shape = img.shape

def butterworth(D0, n, x, y, shape):
    H = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i== x and j==y:
                H[i,j] = 0
            else:
                dist = ((x - i)**2 + (y - j)**2)
                H[i,j] = 1 - (1/(1 + pow(dist/(D0**2), 2/n))) 
            H = np.array(H)
    return H

d0 = 40
H1 = butterworth(d0,1,34,49,shape)
H2 = butterworth(d0,1,98,49,shape)
H3 = butterworth(d0,1,164,49,shape)
H4 = butterworth(d0,1,228,49,shape)
H5 = butterworth(d0,1,295,49,shape)
H6 = butterworth(d0,1,34,147,shape)
H7 = butterworth(d0,1,97,147,shape)
H8 = butterworth(d0,1,291,147,shape)
H9 = butterworth(d0,1,35,245,shape)
H10 = butterworth(d0,1,163,245,shape)
H11 = butterworth(d0,1,293,245,shape)
H12 = butterworth(d0,1,34,342,shape)
H13 = butterworth(d0,1,97,342,shape)
H14 = butterworth(d0,1,229,342,shape)
H15 = butterworth(d0,1,292,342,shape)
H16 = butterworth(d0,1,35,440,shape)
H17 = butterworth(d0,1,98,440,shape)
H18 = butterworth(d0,1,162,440,shape)
H19 = butterworth(d0,1,228,440,shape)
H20 = butterworth(d0,1,290,440,shape)

Htotal = H1*H2*H3*H4*H5*H6*H7*H8*H9*H10*H11*H12*H13*H14*H15*H16*H17*H18*H19*H20
filt_img = fshift*Htotal

"""loc = np.array([(34,49),(98,49),(164,49),(228,49),(295,49),(34,147),(97,147),
(291,147),(35,245),(163,245),(293,245),(34,342),(97,342),(229,342),(292,342),
(35,440),(98,440),(162,440),(228,440),(290,440)])"""


plt.plot(), plt.imshow(np.log(1+np.abs(filt_img)), cmap='gray'),plt.show()