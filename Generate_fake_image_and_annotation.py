import cv2
import numpy as np
import imutils


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




background_path = "G:\\RSA_GIT\\RSA_TomTom\\Fake_database\\background\\946_2__Children_Sign_41.261951700000004_-95.884796.png"
sign_path = "G:\\RSA_GIT\\RSA_TomTom\\Fake_database\\signs\\80 mph\\80mph_01.jpg"

background = cv2.imread(background_path)
sign = cv2.imread(sign_path)


rows,cols,channels = sign.shape
print('rows',rows)
w,h,_ = background.shape
# Let's put the sign on x, y position
x_start = 100
y_start = 100


resize_ratio = 0.1
rotate_degree = 50
rotated = imutils.rotate_bound(sign, rotate_degree)
rows,cols,channels = rotated.shape
print(rotated)

cv2.waitKey(0)

for x in range(rows):
    for y in range(cols):
        if rotated[x][y][0] != 0 and rotated[x][y][1] != 0 and rotated[x][y][2] != 0:
            for i in range(3):
                background[x+x_start][y+y_start][i] = rotated[x][y][i]
cv2.imshow('',background)
cv2.waitKey(0)
