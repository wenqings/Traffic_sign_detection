from PIL import Image
import os
from os import walk
import numpy as np
import cv2
# 0 for signs and 1 for background
# Signs: truncate and png to jpg
# Background: png to jpg
mode = 0
directory = 'G:\\RSA_GIT\\RSA_TomTom\\Fake_database\\signs'

def cut_the_empty_bounding(image):
    rows,cols,c = image.shape
    rows = rows-1
    cols = cols-1
    # ************** top line ********************
    break_flag = False
    for x in range(100,rows):
        if break_flag:
            break
        if x%100 == 0:
            for y in range(cols):
                if image[x][y][0] !=0 or image[x][y][1] !=0 or image[x][y][2] !=0:
                    break_flag = True
                    row_top = x
                    break
    break_flag = False
    for x in range(row_top-100,row_top):
        if break_flag:
            break
        for y in range(cols):
            if image[x][y][0] != 0 or image[x][y][1] != 0 or image[x][y][2] != 0:
                row_top = x
                break_flag = True
                break
    # *************** bottom line *******************
    break_flag = False
    for x in range(0, rows):
        if break_flag:
            break
        if (rows-x) % 100 == 0:
            for y in range(cols):
                if image[rows-x][y][0] != 0 or image[rows-x][y][1] != 0 or image[rows-x][y][2] != 0:
                    break_flag = True
                    row_bottom = rows-x
                    break
    break_flag = False
    for x in range(0,row_bottom):
        if break_flag:
            break
        for y in range(cols):
            if image[rows - x][y][0] != 0 or image[rows - x][y][1] != 0 or image[rows - x][y][2] != 0:
                row_bottom = rows -  x
                break_flag = True
                break
    # ************** left line ********************
    break_flag = False
    for y in range(100,cols):
        if break_flag:
            break
        if y%100 == 0:
            for x in range(rows):
                if image[x][y][0] !=0 or image[x][y][1] !=0 or image[x][y][2] !=0:
                    break_flag = True
                    left_line = y
                    break
    break_flag = False
    for y in range(left_line-100,left_line):
        if break_flag:
            break
        for x in range(rows):
            if image[x][y][0] != 0 or image[x][y][1] != 0 or image[x][y][2] != 0:
                left_line = y
                break_flag = True
                break
    # *************** right line *******************
    break_flag = False
    for y in range(0, cols):
        if break_flag:
            break
        if (cols-y) % 100 == 0:
            for x in range(rows):
                if image[x][cols - y][0] != 0 or image[x][cols - y][1] != 0 or image[x][cols - y][2] != 0:
                    break_flag = True
                    right_line = cols - y
                    break
    break_flag = False
    for y in range(0,right_line):
        if break_flag:
            break
        for x in range(rows):
            if image[x][cols - y][0] != 0 or image[x][cols - y][1] != 0 or image[x][cols - y][2] != 0:
                right_line = cols - y
                break_flag = True
                break


    return image[row_top:row_bottom,left_line:right_line, :]


for (dirpath, dirnames, filenames) in walk(directory):
    for i in range(len(filenames)):
        if filenames[i].endswith('.png') or filenames[i].endswith('.PNG'):
            im = Image.open(os.path.join(dirpath,filenames[i]))
            rgb_im = np.array(im.convert('RGB'))
            if mode == 0:
                try:
                    # new_jpg = cut_the_empty_bounding(rgb_im)
                    new_jpg = im
                    cv2.imwrite(os.path.join(dirpath,filenames[i].replace('.png','.jpg').replace('.PNG','.jpg')),new_jpg)
                    print(os.path.join(dirpath,filenames[i].replace('.png','.jpg').replace('.PNG','.jpg')))
                except:
                    pass
            else:
                try:
                    im.save(os.path.join(dirpath,filenames[i].replace('.png','.jpg').replace('.PNG','.jpg')))
                    print(os.path.join(dirpath,filenames[i].replace('.png','.jpg').replace('.PNG','.jpg')))
                except:
                    pass
