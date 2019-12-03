from image_transformer import ImageTransformer
import sys
import os
import cv2
import numpy as np

image_path = "G:\\RSA_GIT\\RSA_TomTom\\Fake_database\\signs\\fgdbhg.PNG"

h,w,_ = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).shape
img_shape = (h,w)
it = ImageTransformer(image_path, img_shape)


for i in range(1000):
    # theta define the degree rotate on x axis
    # phi define the degree rotate on y axis
    # gamma define the degree rotate on z axis
    rotated_img = it.rotate_along_axis(theta=np.random.normal(0, 10),phi=np.random.normal(0, 30),gamma=np.random.normal(0, 3), dx=h / 2, dy=w / 2)
    cv2.imshow('',rotated_img)
    cv2.waitKey(0)