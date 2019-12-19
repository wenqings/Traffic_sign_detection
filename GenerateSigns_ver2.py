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
import traceback

# Class to store all information needed for XML generation
class XMLPackage:
    def __init__(self, path, filename, width, height, depth, name,
                 xmin, ymin, xmax, ymax):
        self.path = path
        self.filename = filename
        self.width = width
        self.height = height
        self.depth = depth
        # From here, these data are list
        self.name = name
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def generate_XML_File(file_path, data_package):
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = str('train')

    filename = ET.SubElement(annotation, "filename")
    filename.text = str(data_package.filename)

    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    height = ET.SubElement(size, "height")
    depth = ET.SubElement(size, "depth")
    width.text = str(data_package.width)
    height.text = str(data_package.height)
    depth.text = str(data_package.depth)

    for i in range(len(data_package.name)):
        object = ET.SubElement(annotation, "object")
        name = ET.SubElement(object, "name")
        bndbox = ET.SubElement(object, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        ymin = ET.SubElement(bndbox, "ymin")
        xmax = ET.SubElement(bndbox, "xmax")
        ymax = ET.SubElement(bndbox, "ymax")

        name.text = str(data_package.name[i])
        xmin.text = str(data_package.xmin[i])
        ymin.text = str(data_package.ymin[i])
        xmax.text = str(data_package.xmax[i])
        ymax.text = str(data_package.ymax[i])

    XML_data = ET.tostring(annotation)
    file = open(file_path, "w")
    file.write(str(XML_data)[2:len(str(XML_data)) - 1])


background_path = "G:\\RSA_GIT\\RSA_TomTom\\Traffic_sign_detection_Machine_learning\\background\\"
sign_path = "G:\\RSA_GIT\\RSA_TomTom\\Traffic_sign_detection_Machine_learning\\signs\\"
output_path = "T:\\RSA\\output\\train"

background_list = [name for name in os.listdir(background_path)]
background_len = len(background_list) - 1

# Sign type, file name pair
sign_list = []
for r, d, f in os.walk(sign_path):
    for file in f:
        sign_list.append([r.replace('\\', '/').split('/')[-1], file])
sign_len = len(sign_list) - 1
print(sign_list)


def get_random_background():
    return background_path + background_list[random.randint(0, background_len)]


def get_random_sign():
    while True:
        rand = random.randint(0, sign_len)
        if 'speed_limit' != sign_list[rand][0] and 'mph' != sign_list[rand][0]:
            return sign_path + sign_list[rand][0] + '/' + sign_list[rand][1], rand


def get_random_speed_limit():
    while True:
        rand = random.randint(0, sign_len)
        if 'speed_limit' == sign_list[rand][0]:
            return sign_path + sign_list[rand][0] + '/' + sign_list[rand][1], rand


def get_random_mph():
    while True:
        rand = random.randint(0, sign_len)
        if 'mph' == sign_list[rand][0]:
            return sign_path + sign_list[rand][0] + '/' + sign_list[rand][1], rand


def resize_sign(sign):
    height, width, channels = sign.shape
    if width < 30 or height < 30:
        if width<=height:
            resizedImage = cv2.resize(sign, (30, int(30/width*height)+1))
            return resizedImage
        else:

            resizedImage = cv2.resize(sign, (int(30/height*width)+1, 30))
            return resizedImage
    # cv2.imshow('',sign)
    # cv2.waitKey(0)
    # print('sign shape: width:',width,' height:',height)
    # resize sign image
    # width: mean 250 pixel, 10 standard deviation
    new_W = np.random.normal(250, 10)
    # We don't make the image bigger, as it will bring some noise
    # which the noise is different than the camera noise
    # If we want to bring any noise, the noise feature should also match the camera behaviour
    # Arbitrary adding noise will cause image attack/defense issue
    if new_W > width:
        return sign
    # if random size too small, we random again
    if new_W < 30:
        new_W = np.random.normal(250, 10)
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
    row_top = 10
    for x in range(10, rows):
        if break_flag:
            break
        if x % 10 == 0:
            for y in range(cols):
                if image[x][y][0] != 0 or image[x][y][1] != 0 or image[x][y][2] != 0:
                    break_flag = True
                    row_top = x
                    break
    break_flag = False
    for x in range(row_top - 10, row_top):
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
        if (rows - x) % 10 == 0:
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
    left_line = 10
    for y in range(10, cols):
        if break_flag:
            break
        if y % 10 == 0:
            for x in range(rows):
                if image[x][y][0] != 0 or image[x][y][1] != 0 or image[x][y][2] != 0:
                    break_flag = True
                    left_line = y
                    break
    break_flag = False
    for y in range(left_line - 10, left_line):
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
        if (cols - y) % 10 == 0:
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
    image_list = []
    random_bright = .5 + np.random.uniform()
    for i in range(len(image)):
        image1 = cv2.cvtColor(image[i], cv2.COLOR_RGB2HSV)
        image1 = np.array(image1, dtype=np.float64)
        image1[:, :, 2] = image1[:, :, 2] * random_bright
        image1[:, :, 2][image1[:, :, 2] > 255] = 255
        image1 = np.array(image1, dtype=np.uint8)
        image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2RGBA)
        image_list.append(image1)
    return image_list


# Shadow augmentation
def add_random_shadow(image):
    image_list = []
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    for i in range(len(image)):
        image_hls = cv2.cvtColor(image[i], cv2.COLOR_RGB2HLS, cv2.IMREAD_UNCHANGED)
        shadow_mask = 0 * image_hls[:, :, 1]
        X_m = np.mgrid[0:image[i].shape[0], 0:image[i].shape[1]][0]
        Y_m = np.mgrid[0:image[i].shape[0], 0:image[i].shape[1]][1]
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
        image[i] = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
        image[i] = cv2.cvtColor(image[i], cv2.COLOR_RGB2RGBA)
        image_list.append(image[i])
    return image_list


def overlap(image):
    path = 'G:\\RSA_GIT\\RSA_TomTom\\Traffic_sign_detection_Machine_learning\\Augmentation\\overlap\\'
    style_img = random.choice(os.listdir(path))
    style_img = cv2.imread(path + style_img, cv2.IMREAD_UNCHANGED)
    image_list = []
    for i in range(len(image)):
        h, w, c = image[i].shape
        style_img = cv2.resize(style_img, (w, h))
        for row in range(h):
            for col in range(w):
                if style_img[row][col][3] > 100:
                    for c in range(3):
                        image[i][row][col][c] = style_img[row][col][c]
        image[i] = cv2.cvtColor(image[i], cv2.COLOR_RGB2RGBA)
        image_list.append(image[i])
    return image_list


def blend(image):
    path = 'G:\\RSA_GIT\\RSA_TomTom\\Traffic_sign_detection_Machine_learning\\Augmentation\\blend\\'
    style_img = random.choice(os.listdir(path))
    style_img = cv2.imread(path + style_img, cv2.IMREAD_UNCHANGED)
    image_list = []
    for i in range(len(image)):
        h, w, c = image[i].shape
        style_img = cv2.resize(style_img, (w, h))
        rand = np.random.normal(0.3, 0.1)
        if rand < 0:
            rand = 0
        if rand > 0.5:
            rand = 0.5
        img = cv2.addWeighted(image[i], 1 - rand, style_img, rand, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        image_list.append(img)
    return image_list


def generateComposite(composite_count):
    iterator = 46132
    while iterator < composite_count:
        background = cv2.imread(get_random_background(), cv2.IMREAD_UNCHANGED)
        background = background[200:915, 100:1800, :]
        height, width, _ = background.shape
        rand_sign, sign_index = get_random_sign()
        # Each auto-generated image may have multiple signs in it
        name = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        print(iterator)
        # For speed limit sign, here 60% of them we add speed limit word, (80% word on top, 20% word below)
        # 40% add mph word(80% below, 20% right)
        if re.match(r'\d\d\smph', sign_list[sign_index][0]):
            sign = cv2.imread(rand_sign, cv2.IMREAD_UNCHANGED)
            resizedImage = resize_sign(sign)
            if random.random() >= 0.40:
                speed_limit, concat_index = get_random_speed_limit()
                speed_limit = cv2.imread(speed_limit, cv2.IMREAD_UNCHANGED)
                resized_speed_limit = resize_sign(speed_limit)
                augmented_concat = resized_speed_limit
                concate_type = 'speed_limit'
            else:
                mph, concat_index = get_random_mph()
                mph = cv2.imread(mph, cv2.IMREAD_UNCHANGED)
                resized_speed_limit = resize_sign(mph)
                augmented_concat = resized_speed_limit
                concate_type = 'mph'
            # TODO: we can add km/h and so on

            augmented_img = resizedImage

            augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2RGBA)
            augmented_concat = cv2.cvtColor(augmented_concat, cv2.COLOR_RGB2RGBA)
            if random.random() > 0.90:
                img_list = augment_brightness_camera_images([augmented_img, augmented_concat])
                augmented_img = img_list[0]
                augmented_concat = img_list[1]
            if random.random() > 0.90:
                img_list = add_random_shadow([augmented_img, augmented_concat])
                augmented_img = img_list[0]
                augmented_concat = img_list[1]
            if random.random() > 0.90:
                img_list = overlap([augmented_img, augmented_concat])
                augmented_img = img_list[0]
                augmented_concat = img_list[1]
            if random.random() > 0.90:
                img_list = blend([augmented_img, augmented_concat])
                augmented_img = img_list[0]
                augmented_concat = img_list[1]

            h, w, c = augmented_img.shape
            hc, wc, cc = augmented_concat.shape
            # cv2.imshow('1', augmented_concat)
            # cv2.waitKey(0)
            if random.random() > 0.20:
                if c == 4:
                    it = ImageTransformer(augmented_img, (h, w))
                    itc = ImageTransformer(augmented_concat, (hc, wc))
                    theta = np.random.normal(0, 10)
                    if theta < -20 or theta > 20:
                        theta = np.random.normal(0, 1)

                    phi = np.random.normal(0, 30)
                    if phi < -45 or phi > 45:
                        phi = np.random.normal(0, 1)

                    gamma = np.random.normal(0, 3)
                    if gamma < -20 or gamma > 20:
                        gamma = np.random.normal(0, 1)

                    augmented_img = it.rotate_along_axis(theta=theta, phi=phi,
                                                   gamma=gamma, dx=h / 2, dy=w / 2)
                    augmented_concat = itc.rotate_along_axis(theta=theta, phi=phi,
                                                     gamma=gamma, dx=hc / 2, dy=wc / 2)
                    # 3D rotate function somehow resize the image(swap the width and height), I don't know how to fix
                    # in the matrix, so I have to resize it back here
                    augmented_img = cv2.resize(augmented_img, (w, h))
                    augmented_img = cut_the_empty_bounding(augmented_img)
                    augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2RGBA)

                    augmented_concat = cv2.resize(augmented_concat, (wc, hc))
                    augmented_concat = cut_the_empty_bounding(augmented_concat)
                    augmented_concat = cv2.cvtColor(augmented_concat, cv2.COLOR_RGB2RGBA)

            aug_h, aug_w, _ = augmented_img.shape
            aug_concat_h, aug_concat_w, _ = augmented_concat.shape

            # Let's put the sign on x, y position
            x_start_max = int(width) - aug_w - aug_concat_w
            y_start_max = int(height) - aug_h - aug_concat_h

            x_start = random.randint(0, x_start_max)
            y_start = random.randint(aug_concat_h, y_start_max)
            xmin.append(x_start)
            ymin.append(y_start)
            xmax.append(x_start + aug_w)
            ymax.append(y_start + aug_h)
            for x in range(aug_w):
                for y in range(aug_h):
                    if augmented_img[y][x][3] > 50:
                        for i in range(3):
                            background[y + y_start][x + x_start][i] = augmented_img[y][x][i]
            # After we put the sign position, we concatenate the word to it
            if concate_type == 'speed_limit':
                if random.random() >= 0.20:
                    # 80% cases we put speed limit sign above the number
                    y_start = y_start - aug_concat_h
                    xmin.append(x_start)
                    ymin.append(y_start)
                    xmax.append(x_start + aug_concat_w)
                    ymax.append(y_start + aug_concat_h)
                    for x in range(aug_concat_w):
                        for y in range(aug_concat_h):
                            if augmented_concat[y][x][3] > 10:
                                for i in range(3):
                                    background[y + y_start][x + x_start][i] = augmented_concat[y][x][i]
                else:
                    # 20% cases we put speed limit sign below the number
                    y_start = y_start + aug_h
                    xmin.append(x_start)
                    ymin.append(y_start)
                    xmax.append(x_start + aug_concat_w)
                    ymax.append(y_start + aug_concat_h)
                    for x in range(aug_concat_w):
                        for y in range(aug_concat_h):
                            if augmented_concat[y][x][3] > 10:
                                for i in range(3):
                                    background[y + y_start][x + x_start][i] = augmented_concat[y][x][i]

            elif concate_type == 'mph':
                if random.random() >= 0.40:
                    # 60% cases we put mph sign below the number
                    y_start = y_start + aug_h
                    xmin.append(x_start)
                    ymin.append(y_start)
                    xmax.append(x_start + aug_concat_w)
                    ymax.append(y_start + aug_concat_h)
                    for x in range(aug_concat_w):
                        for y in range(aug_concat_h):
                            if augmented_concat[y][x][3] > 10:
                                for i in range(3):
                                    background[y + y_start][x + x_start][i] = augmented_concat[y][x][i]
                else:
                    # 40% cases we put mph right to the number
                    x_start = x_start + aug_w
                    xmin.append(x_start)
                    ymin.append(y_start)
                    xmax.append(x_start + aug_concat_w)
                    ymax.append(y_start + aug_concat_h)
                    for x in range(aug_concat_w):
                        for y in range(aug_concat_h):
                            if augmented_concat[y][x][3] > 50:
                                for i in range(3):
                                    background[y + y_start][x + x_start][i] = augmented_concat[y][x][i]

            cv2.imwrite(output_path + "/" + "Composite" + str(iterator) + ".png", background)
            name.append(sign_list[sign_index][0])
            name.append(sign_list[concat_index][0])

            XML_data = XMLPackage(path=rand_sign.replace('\\', '/'), filename="Composite" + str(iterator) + ".png",
                                  width=width,
                                  height=height, depth="4", name=name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
            generate_XML_File((output_path + "/" + "Composite" + str(iterator) + ".xml"), XML_data)

            iterator = iterator + 1

        else:
            sign = cv2.imread(rand_sign, cv2.IMREAD_UNCHANGED)
            resizedImage = resize_sign(sign)
            augmented_img = resizedImage
            augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2RGBA)

            if random.random() > 0.90:
                augmented_img = augment_brightness_camera_images([augmented_img])[0]
            if random.random() > 0.90:
                augmented_img = add_random_shadow([augmented_img])[0]
            if random.random() > 0.90:
                augmented_img = overlap([augmented_img])[0]
            if random.random() > 0.90:
                augmented_img = blend([augmented_img])[0]


            h, w, c = augmented_img.shape
            # cv2.imshow('1', augmented_concat)
            # cv2.waitKey(0)
            if random.random() > 0.20:
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

                    augmented_img = it.rotate_along_axis(theta=theta, phi=phi,
                                                         gamma=gamma, dx=h / 2, dy=w / 2)
                    # 3D rotate function somehow resize the image(swap the width and height), I don't know how to fix
                    # in the matrix, so I have to resize it back here
                    augmented_img = cv2.resize(augmented_img, (w, h))
                    augmented_img = cut_the_empty_bounding(augmented_img)
                    augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2RGBA)

            aug_h, aug_w, _ = augmented_img.shape

            # Let's put the sign on x, y position
            x_start_max = int(width) - aug_w
            y_start_max = int(height) - aug_h

            x_start = random.randint(0, x_start_max)
            y_start = random.randint(0, y_start_max)
            xmin.append(x_start)
            ymin.append(y_start)
            xmax.append(x_start + aug_w)
            ymax.append(y_start + aug_h)
            for x in range(aug_w):
                for y in range(aug_h):
                    if augmented_img[y][x][3] > 50:
                        for i in range(3):
                            background[y + y_start][x + x_start][i] = augmented_img[y][x][i]
            cv2.imwrite(output_path + "/" + "Composite" + str(iterator) + ".png", background)
            name.append(sign_list[sign_index][0])

            XML_data = XMLPackage(path=rand_sign.replace('\\', '/'), filename="Composite" + str(iterator) + ".png",
                                  width=width,
                                  height=height, depth="4", name=name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
            generate_XML_File((output_path + "/" + "Composite" + str(iterator) + ".xml"), XML_data)

            iterator = iterator + 1


def main():
    generateComposite(100000)


main()
