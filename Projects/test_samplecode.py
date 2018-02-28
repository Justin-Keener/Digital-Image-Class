import cv2 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

img_path = '/Users/justinkeener/Desktop/Digital-Image-Class/images/mountain.JPG'
img = cv2.imread(img_path)
RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
color1 = ('r','g','b')

bins = 32

for i, col in enumerate(color1):
    hist = cv2.calcHist([img], [i], None, [0, 256] * 3)
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122), plt.plot(hist, color= col),plt.xlim([0,256])