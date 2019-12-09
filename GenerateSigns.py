import os
import random
from tkinter import filedialog
from tkinter import *

import cv2
from PIL import Image
import os
from os import walk
import numpy as np
import imutils
import math
from xml.dom import minidom
import xml.etree.ElementTree as ET
from image_transformer import ImageTransformer


# Class to store all information needed for XML generation
class XMLPackage:
    def __init__(self, path, filename, database, width, height, depth, segmented, name, pose, truncated, difficult,
                 xmin, ymin, xmax, ymax):
        self.path = path
        self.filename = filename
        self.database = database
        self.width = width
        self.height = height
        self.depth = depth
        self.segmented = segmented
        self.name = name
        self.pose = pose
        self.truncated = truncated
        self.difficult = difficult
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def generate_XML_File(file_path, data_package):
    annotation = ET.Element("annotation")
    folder = ET.SubElement(annotation, "folder")
    filename = ET.SubElement(annotation, "filename")
    path = ET.SubElement(annotation, "path")
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    height = ET.SubElement(size, "height")
    depth = ET.SubElement(size, "depth")
    segmented = ET.SubElement(annotation, "segmented")
    object = ET.SubElement(annotation, "object")
    name = ET.SubElement(object, "name")
    pose = ET.SubElement(object, "pose")
    truncated = ET.SubElement(object, "truncated")
    difficult = ET.SubElement(object, "difficult")
    bndbox = ET.SubElement(object, "bndbox")
    xmin = ET.SubElement(bndbox, "xmin")
    ymin = ET.SubElement(bndbox, "ymin")
    xmax = ET.SubElement(bndbox, "xmax")
    ymax = ET.SubElement(bndbox, "ymax")

    folder.text = str('train')
    filename.text = str(data_package.filename)
    path.text = str(data_package.path)
    database.text = str(data_package.database)
    width.text = str(data_package.width)
    height.text = str(data_package.height)
    depth.text = str(data_package.depth)
    segmented.text = str(data_package.segmented)
    name.text = str(data_package.name)
    pose.text = str(data_package.pose)
    truncated.text = str(data_package.truncated)
    difficult.text = str(data_package.difficult)
    xmin.text = str(data_package.xmin)
    ymin.text = str(data_package.ymin)
    xmax.text = str(data_package.xmax)
    ymax.text = str(data_package.ymax)

    XML_data = ET.tostring(annotation)
    file = open(file_path, "w")
    file.write(str(XML_data)[2:len(str(XML_data)) - 1])


background_path = "D:\\RSA\\Traffic_sign_detection_Machine_learning\\background\\"
sign_path = "D:\\RSA\\Traffic_sign_detection_Machine_learning\\signs_new\\"
output_path = "D:\\RSA\\output\\test"

background_list = [name for name in os.listdir(background_path)]
background_len = len(background_list) - 1

# Sign type, file name pair
sign_list = []
for r, d, f in os.walk(sign_path):
    for file in f:
        sign_list.append([r.replace('\\', '/').split('/')[-1], file])
sign_len = len(sign_list) - 1


def get_random_background():
    return background_path + background_list[random.randint(0, background_len)]


def get_random_sign():
    rand = random.randint(0, sign_len)
    return sign_path + sign_list[rand][0] + '/' + sign_list[rand][1], rand


def resize_sign(sign):
    height, width, channels = sign.shape
    # cv2.imshow('',sign)
    # cv2.waitKey(0)
    # print('sign shape: width:',width,' height:',height)
    # resize sign image
    # width: mean 150 pixel, 10 standard deviation
    new_W = np.random.normal(250, 10)
    # We don't make the image bigger, as it will bring some noise
    # which the noise is different than the camera noise
    # If we want to bring any noise, the noise feature should also match the camera behaviour
    # Arbitrary adding noise will cause image attack/defense issue
    if new_W > width:
        return sign
    # if random size too small, we random again
    if new_W < 30:
        new_W = np.random.normal(30, 10)
        if new_W < 30:
            new_W = 30

    new_H = height * new_W / width
    resizedImage = cv2.resize(sign, (int(new_W), int(new_H)))
    return resizedImage


def cut_the_empty_bounding(image):
    rows, cols, c = image.shape
    rows = rows - 1
    cols = cols - 1
    # ************** top line ********************
    break_flag = False
    row_top = 50
    for x in range(50, rows):
        if break_flag:
            break
        if x % 50 == 0:
            for y in range(cols):
                if image[x][y][0] != 0 or image[x][y][1] != 0 or image[x][y][2] != 0:
                    break_flag = True
                    row_top = x
                    break
    break_flag = False
    for x in range(row_top - 50, row_top):
        if break_flag:
            break
        for y in range(cols):
            if image[x][y][0] != 0 or image[x][y][1] != 0 or image[x][y][2] != 0:
                row_top = x
                break_flag = True
                break
    # *************** bottom line *******************
    break_flag = False
    row_bottom = rows
    for x in range(0, rows):
        if break_flag:
            break
        if (rows - x) % 50 == 0:
            for y in range(cols):
                if image[rows - x][y][0] != 0 or image[rows - x][y][1] != 0 or image[rows - x][y][2] != 0:
                    break_flag = True
                    row_bottom = rows - x
                    break
    break_flag = False
    for x in range(0, row_bottom):
        if break_flag:
            break
        for y in range(cols):
            if image[rows - x][y][0] != 0 or image[rows - x][y][1] != 0 or image[rows - x][y][2] != 0:
                row_bottom = rows - x
                break_flag = True
                break
    # ************** left line ********************
    break_flag = False
    left_line = 50
    for y in range(50, cols):
        if break_flag:
            break
        if y % 50 == 0:
            for x in range(rows):
                if image[x][y][0] != 0 or image[x][y][1] != 0 or image[x][y][2] != 0:
                    break_flag = True
                    left_line = y
                    break
    break_flag = False
    for y in range(left_line - 50, left_line):
        if break_flag:
            break
        for x in range(rows):
            if image[x][y][0] != 0 or image[x][y][1] != 0 or image[x][y][2] != 0:
                left_line = y
                break_flag = True
                break
    # *************** right line *******************
    break_flag = False
    right_line = cols
    for y in range(0, cols):
        if break_flag:
            break
        if (cols - y) % 50 == 0:
            for x in range(rows):
                if image[x][cols - y][0] != 0 or image[x][cols - y][1] != 0 or image[x][cols - y][2] != 0:
                    break_flag = True
                    right_line = cols - y
                    break
    break_flag = False
    for y in range(0, right_line):
        if break_flag:
            break
        for x in range(rows):
            if image[x][cols - y][0] != 0 or image[x][cols - y][1] != 0 or image[x][cols - y][2] != 0:
                right_line = cols - y
                break_flag = True
                break

    return image[row_top:row_bottom, left_line:right_line, :]


# Brightness augmentation
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2RGBA)
    return image1


# Shadow augmentation
def add_random_shadow(image):
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS, cv2.IMREAD_UNCHANGED)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    # random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    return image


def overlap(image):
    path = 'G:\\RSA_GIT\\RSA_TomTom\\Fake_database\\Augmentation\\overlap\\'
    style_img = random.choice(os.listdir(path))
    style_img = cv2.imread(path + style_img, cv2.IMREAD_UNCHANGED)
    h, w, c = image.shape
    style_img = cv2.resize(style_img, (w, h))
    for row in range(h):
        for col in range(w):
            if style_img[row][col][3] > 100:
                for i in range(3):
                    image[row][col][i] = style_img[row][col][i]
    # print('overlap')

    return image


def blend(image):
    path = 'G:\\RSA_GIT\\RSA_TomTom\\Fake_database\\Augmentation\\blend\\'
    style_img = random.choice(os.listdir(path))
    style_img = cv2.imread(path + style_img, cv2.IMREAD_UNCHANGED)
    h, w, c = image.shape
    style_img = cv2.resize(style_img, (w, h))
    rand = np.random.normal(0.3, 0.1)
    if rand < 0:
        rand = 0
    if rand > 0.5:
        rand = 0.5
    img = cv2.addWeighted(image, 1 - rand, style_img, rand, 0)
    return img


def generateComposite(composite_count):
    iterator = 0

    while iterator < composite_count:

        background = cv2.imread(get_random_background(), cv2.IMREAD_UNCHANGED)
        background = background[200:915, 100:1800, :]
        rand_sign, index = get_random_sign()
        sign = cv2.imread(rand_sign, cv2.IMREAD_UNCHANGED)
        height, width, _ = background.shape
        # sign = cut_the_empty_bounding(sign)
        resizedImage = resize_sign(sign)
        augmented_img = resizedImage
        if random.random() > 0.90:
            augmented_img = augment_brightness_camera_images(augmented_img)
        if random.random() > 0.90:
            augmented_img = add_random_shadow(augmented_img)
        if random.random() > 0.90:
            augmented_img = overlap(augmented_img)
        if random.random() > 0.90:
            augmented_img = blend(augmented_img)
        h, w, c = augmented_img.shape

        if c == 4:
            it = ImageTransformer(augmented_img, (h, w))
            theta = np.random.normal(0, 10)
            if theta < -20 or theta > 20:
                theta = np.random.normal(0, 1)

            phi = np.random.normal(0, 30)
            if phi < -45 or phi > 45:
                phi = np.random.normal(0, 1)

            gamma = np.random.normal(0, 3)
            if gamma < -20 or gamma > 20:
                gamma = np.random.normal(0, 1)

            rotated = it.rotate_along_axis(theta=theta, phi=phi,
                                           gamma=gamma, dx=h / 2, dy=w / 2)
            # 3D rotate function somehow resize the image(swap the width and height), I don't know how to fix
            # in the matrix, so I have to resize it back here
            rotated = cv2.resize(rotated, (w, h))
            rotated = cut_the_empty_bounding(rotated)
            rotated = cv2.cvtColor(rotated, cv2.COLOR_RGB2RGBA)

            rotated_h, rotated_w, _ = rotated.shape

            # Let's put the sign on x, y position

            # Extremely inconsistent results with x and y offset ?????? (spent lots of time debugging)
            x_start_max = int(width) - rotated_w
            y_start_max = int(height) - rotated_h

            x_start = random.randint(0, x_start_max)
            y_start = random.randint(0, y_start_max)
            for x in range(rotated_w):
                for y in range(rotated_h):

                    if rotated[y][x][3] > 50:
                        for i in range(3):
                            background[y + y_start][x + x_start][i] = rotated[y][x][i]

            cv2.imwrite(output_path + "/" + "Composite" + str(iterator) + ".png", background)

            XML_data = XMLPackage(rand_sign.replace('\\', '/'), "Composite" + str(iterator) + ".png", "Unknown", width,
                                  height, "4", "0", sign_list[index][0], "Unspecified", "0", "0", int(x_start),
                                  int(y_start), int(x_start + rotated_w), int(y_start + rotated_h))
            generate_XML_File((output_path + "/" + "Composite" + str(iterator) + ".xml"), XML_data)

            iterator = iterator + 1
            print(iterator)
        else:
            print(rand_sign)


def main():
    generateComposite(2000)


main()
