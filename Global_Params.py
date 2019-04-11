#!/usr/bin/env python
#-*- coding: utf-8 -*-

###########################
# [Global Parameters] etc #
###########################

################################ CNN + LSTM Video 2 Text #####################################
## N-Frames
num_frames = 20 # 80 40 20 10 train:40 test:20

## Img Size
img_height = 540 # 512
img_width = 960 # 512

## Train Path
video_path_train = '/home/thomas/project/video_to_sequence-master/data/train_videos'
video_feat_path_train = '/home/thomas/project/video_to_sequence-master/data/train_features'

## Test Path
video_path_test = '/home/thomas/project/video_to_sequence-master/data/test_videos'
video_feat_path_test = '/home/thomas/project/video_to_sequence-master/data/test_features'



## Label Data
# video_data_path = '/home/thomas/project/video_to_sequence-master/data/video_corpus.csv'
# video_data_path = '/home/thomas/project/video_to_sequence-master/data/video_corpus_(frames).csv'
# video_data_path = '/home/thomas/project/video_to_sequence-master/data/video_corpus_(alex 0-1).csv'
# video_data_path = '/home/thomas/project/video_to_sequence-master/data/video_corpus_(alex 2).csv'
video_data_path = '/home/thomas/project/video_to_sequence-master/data/video_corpus_(alex 3).csv'

## Confidence List
confidence_path = '/home/thomas/project/video_to_sequence-master/data/confidence_list'

## Real-Time Frames
RT_frames_save_path = '/home/thomas/project/video_to_sequence-master/data/realtime_frames'

## Real-Time Features
RT_feats_save_fullpath = '/home/thomas/project/video_to_sequence-master/data/realtime_features/RT_Feats.npy'

## Model
model_folder = '/home/thomas/project/video_to_sequence-master/models'
model_path = '/home/thomas/project/video_to_sequence-master/models/model-900'
vgg16_path = '/home/thomas/caffe/models/tensorflow_vgg16/vgg16.tfmodel'

## Keywords List
'''
keywords_list = [['nobody', 'doing', 'nothing'],
                ['person', 'wiping', 'table']]
# '''
# '''
keywords_list = [['nobody', 'doing', 'nothing'],
                ['person', 'wiping', 'table'],
                ['person', 'opening', 'door']]
# '''
################################ CNN + LSTM Video 2 Text #####################################

####################################### ROS Topics ###########################################
image_topic = '/kinect2/sd/image_color_rect' # 512 * 424
# image_topic = '/kinect2/qhd/image_color' # 960 * 540
# image_topic = '/kinect2/hd/image_color' # 1920 * 1080
pointcloud_topic = '/kinect2/sd/points' # 512 * 424 Better
# pointcloud_topic = '/kinect2/qhd/points' # 960 * 540
# pointcloud_topic = '/kinect2/hd/points' # 1920 * 1080 Worse
handcamera_topic = '/cameras/right_hand_camera/image'
headcamera_topic = '/cameras/head_camera/image'
kinect_publish_topic = '/casia/kinect2/object_pose'
boardmess_publish_topic = '/casia/kinect2/dirty_point'
boardimage_publish_topic = '/casia/kinect2/board_image'
usbcamera_topic = '/usb_cam/image_raw'
usbcamera_publish_topic = '/zhishan/xy_pose'
# publish_topic = '/image_topic_return'
####################################### ROS Topics ###########################################

################################# Py-Faster-RCNN CLASSES #####################################
## CLASSES list in Py-Faster-RCNN object detection
# ''' 2 + 1 Classes
classes_list = [(1, 'bottle'), (2, 'person')]
# classes_list = [(1, 'bottle')]
# classes_list = [(2, 'person')]
# '''
''' 20 + 1 Classes
classes_list = [(5, 'bottle'), (15, 'person')]
# classes_list = [(5, 'bottle')]
# classes_list = [(15, 'person')]
# '''
''' VOC Datasets 20 CLASSES list
classes_list = [
(1, 'aeroplane'),
(2, 'bicycle'),
(3, 'bird'),
(4, 'boat'),
(5, 'bottle'),
(6, 'bus'),
(7, 'car'),
(8, 'cat'),
(9, 'chair'),
(10, 'cow'),
(11, 'diningtable'),
(12, 'dog'),
(13, 'horse'),
(14, 'motorbike'),
(15, 'person'),
(16, 'pottedplant'),
(17, 'sheep'),
(18, 'sofa'),
(19, 'train'),
(20, 'tvmonitor')]
# '''
################################# Py-Faster-RCNN CLASSES #####################################

################################### HTM Video 2 Text #########################################
## Train Path
HTM_video_path_train = '/home/thomas/project/Pan/Hop_TM/Data/train_videos'
HTM_video_feat_path_train = '/home/thomas/project/Pan/Hop_TM/Data/train_features'

## Test Path
HTM_video_path_test = '/home/thomas/project/Pan/Hop_TM/Data/test_videos'
HTM_video_feat_path_test = '/home/thomas/project/Pan/Hop_TM/Data/test_features'
################################### HTM Video 2 Text #########################################