import cv2
import dlib
import os
import sys 
import numpy as np

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x 
    h = rect.bottom() - y 
    left = int(x + w/2 - 128)
    right = left + 256
    top = 0
    bottom = 256
    return (left, top, right, bottom)


def resize(image, width=456):
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r)) 
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


image_root = '/home/work_nfs3/xswang/data/avatar/Obama/clip/select/images/'
save_root = '/home/work_nfs3/xswang/data/avatar/Obama/clip/select/images_crop/'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/work_nfs3/xswang/data/avatar/Obama/data_processing_code/shape_predictor_68_face_landmarks.dat")

names = os.listdir(image_root)
i = 0
total = len(names)
log_file = 'log/crop_image.text'
for name in names:
    i += 1    
    file_path = os.path.join(image_root,name)
    save_file = os.path.join(save_root,name)
    if not os.path.exists(save_file):
        os.mkdir(save_file)
    image_names = os.listdir(file_path)
    first_path = os.path.join(file_path,image_names[0])
    first_image = cv2.imread(first_path)
    first_image = resize(first_image, width=456)
    gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    rect = detector(gray, 1)
    try:
        x1,y1,x2,y2 =  rect_to_bb(rect[0]) 
        px1,py1,px2,py2 = x1,y1,x2,y2
    except:
        x1,y1,x2,y2 = px1,py1,px2,py2
        pass
    if abs(x1-x2)<10 or abs(y1-y2)<10:
        x1,y1,x2,y2 = px1,py1,px2,py2
    for image_name in image_names:
        save_path = os.path.join(save_file,image_name)
        image_path = os.path.join(file_path,image_name)
        image = cv2.imread(image_path)
        image = resize(image, width=456)
        crop_img = image[y1:y2,x1:x2]
        cv2.imwrite(save_path, crop_img)
    info = 'processed the %d of total %d image'%(i,total)
    print(info)
    # if i % 10 == 0 or i == total:
    #     with open(log_file,'a') as f:
    #         f.writelines(info)