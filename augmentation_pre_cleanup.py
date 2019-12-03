import cv2
import numpy as np
import os
from os import walk


# Run this script for overlap images
# Make the augmentation material background transparent based on the color
# For png, shape: height, width, channel
# white: [255,255,255,_]
# No transparent: 255, transparent: 0
# This script we make white part transparent
# More white it is, more transparent it is


#
# path = 'G:\\RSA_GIT\\RSA_TomTom\\Fake_database\\Augmentation\\overlap'
#
# for (dirpath, dirnames, filenames) in walk(path):
#     for f in filenames:
#         if '.PNG' in f or '.png' in f:
#             print(os.path.join(dirpath,f))
#             img_path = os.path.join(dirpath, f)
#             img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#             h,w,c = img.shape
#             for row in range(h):
#                 for col in range(w):
#                     new_t =int(255-(int(img[row][col][0])+int(img[row][col][1])+int(img[row][col][2]))/3)
#                     img[row][col][3] = int(new_t)
#             cv2.imwrite(img_path, img)



pp = "G:\\RSA_GIT\\RSA_TomTom\\Fake_database\\Augmentation\\ewrghfds.PNG"

img = cv2.imread(pp)
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (0,0,img.shape[1],img.shape[0])

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
cv2.imshow('',img)
cv2.waitKey(0)
