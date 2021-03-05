import cv2
import dlib
import os
import sys 
import numpy as np
from utils import *

detector = dlib.get_frontal_face_detector()
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords
def getTilt(keypoints_mn):
    # Remove in plane rotation using the eyes
    eyes_kp = np.array(keypoints_mn[36:47])
    x = eyes_kp[:, 0]
    y = -1*eyes_kp[:, 1]
    # print('X:', x)
    # print('Y:', y)
    m = np.polyfit(x, y, 1)
    tilt = np.degrees(np.arctan(m[0]))
    return tilt

def getKeypointFeatures(keypoints):
    # Mean Normalize the keypoints wrt the center of the mouth
    # Leads to face position invariancy
    mouth_kp_mean = np.average(keypoints[48:67], 0)
    keypoints_mn = keypoints - mouth_kp_mean

    # Remove tilt
    x_dash = keypoints_mn[:, 0]
    y_dash = keypoints_mn[:, 1]
    theta = np.deg2rad(getTilt(keypoints_mn))
    c = np.cos(theta);	s = np.sin(theta)
    x = x_dash*c - y_dash*s	# x = x'cos(theta)-y'sin(theta)
    y = x_dash*s + y_dash*c # y = x'sin(theta)+y'cos(theta)
    keypoints_tilt = np.hstack((x.reshape((-1,1)), y.reshape((-1,1))))

    # Normalize
    N = np.linalg.norm(keypoints_tilt, 2)
    return [keypoints_tilt/N, N, theta, mouth_kp_mean]
def get_facial_landmarks(image):
#     image = io.imread(filename);
    # detect face(s)
    dets = detector(image, 1);
    shape = np.empty([1,1])
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(image, d);
        shape = shape_to_np(shape);
    return shape;


image_root = '/home/work_nfs3/xswang/data/avatar/Obama/clip/select/images_crop'
save_mask_root = '/home/work_nfs3/xswang/data/avatar/Obama/clip/select/for_ObamaNet/images_mask'
save_drawlip_root = '/home/work_nfs3/xswang/data/avatar/Obama/clip/select/for_ObamaNet/images_lip'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/work_nfs3/xswang/code/model_para/shape_predictor_68_face_landmarks.dat")

names = os.listdir(image_root)
j = 0
for name in names:
    j += 1
    file_path = os.path.join(image_root,name)
    save_mask_file = os.path.join(save_mask_root,name)
    save_draw_file = os.path.join(save_drawlip_root,name)
    if not os.path.exists(save_mask_file):
        os.mkdir(save_mask_file)
    if not os.path.exists(save_draw_file):
        os.mkdir(save_draw_file)
    image_names = os.listdir(file_path)
    for image_name in image_names:
        save_mask_path = os.path.join(save_mask_file,image_name)
        save_draw_path = os.path.join(save_draw_file,image_name.replace('.jpg','.png'))
        image_path = os.path.join(file_path,image_name)
        image = cv2.imread(image_path)
        keypoints = get_facial_landmarks(image)
        if not (keypoints.shape[0] == 1):
            l = getKeypointFeatures(keypoints)
            prev_store_l = l
            prev_store_image = image.copy()
        else:
            l = prev_store_l
            image = prev_store_image
        unit_kp, N, tilt, mean = l[0], l[1], l[2], l[3]
        kp_mouth = unit_kp[48:68]
        # create a patch based on the tilt, mean and the size of face
        mean_x, mean_y = int(mean[0]), int(mean[1])
        size = int(N/15)
        aspect_ratio_mouth = 1.8     
        mean_x, mean_y = int(mean[0]), int(mean[1])
        size = int(N/15)
        aspect_ratio_mouth = 1.8     
        patch_img = image.copy()
        patch_img[ mean_y-size: mean_y+size, mean_x-int(aspect_ratio_mouth*size):mean_x+int(aspect_ratio_mouth*size) ] = 0    

        cv2.imwrite(save_mask_path, patch_img)
        if not (keypoints.shape[0] == 1): 
            drawLips(keypoints, patch_img)
        cv2.imwrite(save_draw_path, patch_img)
        info = 'processing %d of %d videos'%(j,len(names))
        print(info)