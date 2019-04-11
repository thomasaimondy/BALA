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
import Global_Params
import ipdb

########################### Global Parameters ###############################

########################### Global Parameters ###############################

############################ Initialization #################################

############################ Initialization #################################

# process single frame to target height & width
def preprocess_frame(image, target_height = 224, target_width = 224):
    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape

    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

# get features from frames video to *.npy
def get_features_from_frames(video_path, video_save_path):
    # vgg_model = '/home/thomas/caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
    # vgg_deploy = '/home/thomas/caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'
    vgg_model = '/home/bibo/Data/video_to_sequence/caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
    vgg_deploy = '/home/bibo/Data/video_to_sequence/caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'
    cnn = CNN(model = vgg_model, deploy = vgg_deploy, width = 224, height = 224)

    while(True):# real-time loop
        frames_video_list = os.listdir(video_path) # frames-video folder-list
        if len(frames_video_list) == 0:# if empty
            time.sleep(2)
            print time.time(), "  ", "..."
            continue

        for frames_video_name in frames_video_list: # process single frames-video one by one
            print time.time(), "  ", "frames_video_name =", frames_video_name
            if os.path.exists(os.path.join(video_save_path, frames_video_name + '.npy')): # if *.npy already exist
                print '\033[92m' + 'Already processed.' + '\033[0m'
                time.sleep(2)
                continue

            print '\033[93m' + 'Processing ...' + '\033[0m'
            frames_fullpath = os.path.join(video_path, frames_video_name) # frames-folder full-path

            frames = os.listdir(frames_fullpath) # frames image-list
            if len(frames) == 0: # if empty
                time.sleep(2)
                print time.time(), "  ", "..."
                continue

            frame_count = 0
            frame_list = []
            frames.sort(key = lambda x: int(x[1 : -5])) # SORT by time index "(NUM).jpg"
            for frame in frames: # read single frame one by one
                frame_fullpath = os.path.join(frames_fullpath, frame)
                img = cv2.imread(frame_fullpath)
                frame_list.append(img)
                frame_count += 1

            frame_list = np.array(frame_list)

            if frame_count > Global_Params.num_frames: # linear sample to (Global_Params.num_frames) frames
                frame_indices = np.linspace(0, frame_count, num = Global_Params.num_frames, endpoint = False).astype(int)
                frame_list = frame_list[frame_indices] # sampled to (Global_Params.num_frames) frames with same intervals

            cropped_frame_list = np.array(map(lambda x: preprocess_frame(x), frame_list)) # preprocess each frame

            feats = cnn.get_features(cropped_frame_list) # get overall features
            save_full_path = os.path.join(video_save_path, frames_video_name + '.npy')
            np.save(save_full_path, feats)
            print '\033[94m' + 'Saved.' + '\033[0m'
            #return feats

# get continuous features from frames video to multiple *.npy
def get_continuous_features_from_frames(video_path, video_save_path):
    # vgg_model = '/home/thomas/caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
    # vgg_deploy = '/home/thomas/caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'
    vgg_model = '/home/bibo/Data/video_to_sequence/caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
    vgg_deploy = '/home/bibo/Data/video_to_sequence/caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'
    cnn = CNN(model = vgg_model, deploy = vgg_deploy, width = 224, height = 224)

    while(True): # real-time loop
        frames_video_list = os.listdir(video_path) # frames-video folder-list
        if len(frames_video_list) == 0: # if empty
            time.sleep(2)
            print time.time(), "  ", "..."
            continue

        for frames_video_name in frames_video_list: # process single frames-video one by one
            print time.time(), "  ", "frames_video_name =", frames_video_name
            if os.path.exists(os.path.join(video_save_path, frames_video_name)): # if already exist
                print '\033[92m' + 'Already processed.' + '\033[0m'
                time.sleep(2)
                continue

            print '\033[93m' + 'Processing ...' + '\033[0m'
            os.mkdir(os.path.join(video_save_path, frames_video_name)) # else: make target directory
            frames_fullpath = os.path.join(video_path, frames_video_name) # frames-folder full-path

            frames = os.listdir(frames_fullpath) # frames image-list

            if len(frames) == 0: # if empty
                time.sleep(2)
                print time.time(), "  ", "..."
                continue

            frame_count = 0
            frame_list = []
            frames.sort(key = lambda x: int(x[1 : -5])) # SORT by time index "(NUM).jpg"
            for frame in frames: # read single frame one by one
                frame_fullpath = os.path.join(frames_fullpath, frame)
                img = cv2.imread(frame_fullpath)
                frame_list.append(img)
                frame_count += 1

            frame_list = np.array(frame_list)
            cropped_frame_list = np.array(map(lambda x: preprocess_frame(x), frame_list)) # preprocess each frame

            for t in xrange(0, frame_count - Global_Params.num_frames + 1): # N frames -> (N - Global_Params.num_frames + 1) features
                continuous_frame = cropped_frame_list[t : t + Global_Params.num_frames]
                feats = cnn.get_features(continuous_frame) # get continuous features
                save_full_path = os.path.join(video_save_path, frames_video_name, str(t) + '.npy')
                np.save(save_full_path, feats)
                print "feature t =", t, '\033[94m' + 'Saved.' + '\033[0m'
                #return feats

##########################
if __name__ == "__main__":

    mode = 2
    # 0 - TRAIN
    # 1 - TEST
    # 2 - TEST_continuous

    if mode == 0:
        sys.exit(get_features_from_frames(Global_Params.video_path_train, Global_Params.video_feat_path_train))
    elif mode == 1:
        sys.exit(get_features_from_frames(Global_Params.video_path_test, Global_Params.video_feat_path_test))
    elif mode == 2:
        sys.exit(get_continuous_features_from_frames(Global_Params.frames_src_path, Global_Params.frames_feat_path))
    else:
        print 'Mode Error!'