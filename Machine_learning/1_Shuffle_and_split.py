import time
import random, os
import shutil
import traceback

# The file name is the same for image and corresponding annotation
images_path = 'D:\\MTSD\\images\\'
annotation_path = 'D:\\MTSD\\annotations\\'

train_output_path = 'D:\\MTSD\\train\\'
test_output_path = 'D:\\MTSD\\test\\'

l = os.listdir(images_path)
random.shuffle(l)

total_images = len(l)
train_image_number = int(total_images * 0.7)
test_image_number = total_images - train_image_number
# Some image don't have traffic sign, so no annotation
for i in range(total_images):
    if i < train_image_number:
        try:
            shutil.move(images_path + l[i], train_output_path + l[i])
            shutil.move(annotation_path + l[i].split('.')[0]+".json", train_output_path + l[i].split('.')[0]+".json")
        except:
            print(traceback.format_exc())
    else:
        try:
            shutil.move(images_path + l[i], test_output_path + l[i])
            shutil.move(annotation_path + l[i].split('.')[0]+".json", test_output_path + l[i].split('.')[0]+".json")
        except:
            print(traceback.format_stack())
