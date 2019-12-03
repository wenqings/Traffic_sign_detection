import cv2
import numpy as np


# Brightness augmentation
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2RGBA)
    return image1



# Shadow augmentation
def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS,cv2.IMREAD_UNCHANGED)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2RGBA)
    return image


def overlap(image,style_img):
    h,w,c  = image.shape
    style_img = cv2.resize(style_img,(w,h))
    print(image.shape)
    print(style_img.shape)
    for row in range(h):
        for col in range(w):
            if style_img[row][col][3]>100:
                for i in range(3):
                    img[row][col][i] = style_img[row][col][i]
    return img



def blend(image,style_img):
    h,w,c  = image.shape
    style_img = cv2.resize(style_img,(w,h))
    print(image.shape)
    print(style_img.shape)
    img = cv2.addWeighted(image,0.7,style_img,0.3,0)
    return img



sign_path = "G:\\RSA_GIT\\RSA_TomTom\\Fake_database\\signs\\35 mph\\ASDRG.PNG"
img = cv2.imread(sign_path,cv2.IMREAD_UNCHANGED)
print(img.shape)
cv2.imshow('',img)
cv2.waitKey(0)

print('fff')
aug_img = add_random_shadow(img)
cv2.imshow('',aug_img)
cv2.waitKey(0)


aug_img = augment_brightness_camera_images(aug_img)
cv2.imshow('',aug_img)
cv2.waitKey(0)

img = blend(aug_img)
cv2.imshow('',img)
cv2.waitKey(0)


img = overlap(aug_img)
cv2.imshow('',img)
cv2.waitKey(0)