import math
import cv2
import numpy as np

import os
import errno

from os import path

#import assignment8 as a8
import panorama as a8


SRC_FOLDER = "images/source"
num_matches =10


from matplotlib import pyplot as plt

image1 = plt.imread('images/source/sample/1.jpg')
image2 = plt.imread('images/source/sample/2.jpg')
image3 = plt.imread('images/source/sample/3.jpg')


corner_1 = a8.getImageCorners(image1)
corner_2 = a8.getImageCorners(image2)
corner_3 = a8.getImageCorners(image3)


image_1_kp, image_2_kp, matches = a8.findMatchesBetweenImages(image1,image2,num_matches)
#print image_1_kp

homography = a8.findHomography(image_1_kp, image_2_kp, matches)
print homography


min_xy, max_xy = a8.getBoundingCorners(corner_1, corner_2, homography)
print min_xy, max_xy

warped_image = a8.warpCanvas(image1, homography, min_xy, max_xy)
print warped_image
plt.imshow(warped_image)



point = np.int64(-min_xy)
img2 = np.zeros(warped_image.shape)
#img2[:image2.shape[0],:image2.shape[1]] = image2

#img2 = np.zeros(image2.shape)
#img2[:image2.shape[0]][:image2.shape[1]] = image2


img2[point[1]:point[1]+image2.shape[0], point[0]:point[0]+image2.shape[1]] = image2

    #img2[point[1]:point[1]+image_2.shape[0], point[0]:point[0]+image_2.shape[1]] = image_2

img2 = np.uint8(img2)

    
mask = img2.copy()
mask[np.where(mask > 0)] = 1
    #mask = cv2.GaussianBlur(mask, (11, 11), 0.7)

output_image = warped_image * (1 - mask) + img2 * mask    
plt.imshow(output_image)    
plt.imshow(warped_image)    
plt.imshow(img2)    
plt.imshow(image2)
plt.imshow(warped_image)


# to combine with the 3rd image

