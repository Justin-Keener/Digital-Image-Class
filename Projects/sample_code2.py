import cv2 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

img_path = "/Users/justinkeener/Desktop/Digital-Image-Class/images/peacoke feather.JPG"
img = cv2.imread(img_path)
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Histogram Equalization for Colored Images

color1 = ('b','g','r')

channels = cv2.split(RGB_img)
eq_channels = []
for ch, color in zip(channels, ['B', 'G','R']):
    eq_channels.append(cv2.equalizeHist(ch))

eq_img = cv2.merge(eq_channels)
eq_img = cv2.cvtColor(eq_img, cv2.COLOR_BGR2RGB)

for i, col in enumerate(color1):
    hist = cv2.calcHist([img],[i],None,[256],[0,256])
    
    hist2 = cv2.calcHist([eq_img],[i], None, [256], [0,256])
    
    plt.subplot(221), plt.imshow(RGB_img)
    plt.subplot(222), plt.imshow(eq_img)
    plt.subplot(223), plt.plot(hist, color= col),plt.xlim([0,256])
    plt.subplot(224), plt.plot(hist2, color= col),plt.xlim([0,256])

plt.show()
