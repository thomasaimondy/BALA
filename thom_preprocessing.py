#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import skimage
from cnn_util import *
# import Global_Params
import ipdb

########################### Global Parameters ###############################

########################### Global Parameters ###############################

############################ Initialization #################################

############################ Initialization #################################

# process single frame to target height & width
def preprocess_frame(image, target_height = 224, target_width = 224):
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height) / height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:, cropping_length : resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length : resized_image.shape[0] - cropping_length, :]

    return cv2.resize(resized_image, (target_height, target_width))

# get features from *.avi videos to *.npy
def get_features_from_video_filemode(video_path, feat_save_path,num_frames):
    vgg_model = root_path + 'caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
    vgg_deploy = root_path + 'caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'
    cnn = CNN(model = vgg_model, deploy = vgg_deploy, width = 224, height = 224)

    videos = os.listdir(video_path)
    videos = filter(lambda x : x.endswith('avi'), videos) # AVI only
    for video in videos:
        print time.time(), "  ", "video = ", video
        if os.path.exists(os.path.join(feat_save_path, video + '.npy')):
            print('Already processed.')
            return
        video_fullpath = os.path.join(video_path, video)
        try:
            cap = cv2.VideoCapture(video_fullpath)
        except:
            print('video process error')
            return
        frame_count = 0
        frame_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_list.append(frame)
            frame_count += 1
        frame_list = np.array(frame_list)
        if frame_count > num_frames:
            frame_indices = np.linspace(0, frame_count, num = num_frames, endpoint = False).astype(int)
            frame_list = frame_list[frame_indices]
        cropped_frame_list = np.array(map(lambda x: preprocess_frame(x), frame_list))
        feats = cnn.get_features(cropped_frame_list)
        save_full_path = os.path.join(feat_save_path, video + '.npy')
        np.save(save_full_path, feats)
        # print '\033[94m' + 'Saved.' + '\033[0m'
            #return feats

root_path = '/home/bibo/Data/video_to_sequence/'
# tested by thomas 2019-3-28
def get_features_from_frames_filemode(images_path, feat_save_path):
    vgg_model = root_path + 'caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
    vgg_deploy = root_path + 'caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'
    cnn = CNN(model = vgg_model, deploy = vgg_deploy, width = 224, height = 224)
    imagepaths = os.listdir(images_path)
    frame_list = []
    for imagepath in imagepaths:
        image_fullpath = os.path.join(images_path, imagepath)
        image = cv2.imread(image_fullpath) 

        frame_list.append(image)
    nframes = 20
    cropped_frame_list = np.array(map(lambda x: preprocess_frame(x), frame_list))
    feats = cnn.get_features(cropped_frame_list[0:nframes])
    save_full_path = os.path.join(feat_save_path, 'generate.npy')
    np.save(save_full_path, feats)
    print(save_full_path)


##########################
if __name__ == "__main__":
    video_src_path = root_path + 'data/test_one_video/'
    feat_save_path = root_path + 'data/test_one_feature/'
    num_frames = 20
    get_features_from_video_filemode(video_src_path, feat_save_path,num_frames)

    # images_path = root_path + 'data/test_one_video/task1/'
    # feat_save_path = root_path + 'data/test_one_feature/task1/'
    # get_features_from_frames_filemode(images_path, feat_save_path) #generate.npy 