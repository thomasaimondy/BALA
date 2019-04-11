#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os
import sys
import time
import cv2
import numpy as np
import skimage
import rospy
from cnn_util import CNN
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import Global_Params
import ipdb

########################### Global Parameters ###############################
num_frames = Global_Params.num_frames #80
########################### Global Parameters ###############################

############################### Initialization ###################################
# CNN model
# vgg_model = '/home/thomas/caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
# vgg_deploy = '/home/thomas/caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'
vgg_model = '/home/bibo/Data/video_to_sequence/caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
vgg_deploy = '/home/bibo/Data/video_to_sequence/caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'
cnn = CNN(model = vgg_model, deploy = vgg_deploy, width = 224, height = 224)

# Time-Series ndArray
frames_ARRAY = np.zeros((Global_Params.img_height, Global_Params.img_width, 3, num_frames), dtype = np.uint8) # 4-dims
############################### Initialization ###################################

## process single frame to target height & width
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
        resized_image = cv2.resize(image, (int(width * float(target_height) / height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

## get continuous features from frames video
def get_features_from_continuous_frames(frames_array):
    frame_list = []
    for t in xrange(0, num_frames): # acess single frame one by one
        frame_list.append(frames_array[:, :, :, t])

    # ipdb.set_trace()
    frame_list = np.array(frame_list)
    cropped_frame_list = np.array(map(lambda x: preprocess_frame(x), frame_list)) # preprocess each frame

    t_start = time.clock()
    feats = cnn.get_features(cropped_frame_list) # get continuous features
    t_end = time.clock()
    print 'time(get_features) =', t_end - t_start

    np.save(Global_Params.RT_feats_save_fullpath, feats)
    print 'Features saved.'

## Image Converter
class image_converter:
    def __init__(self):
        self.count = 0
        self.bridge = CvBridge()

        # ipdb.set_trace()
        self.image_sub = rospy.Subscriber(Global_Params.image_topic, Image, self.callback)
        # self.image_pub = rospy.Publisher(Global_Params.publish_topic, Image, queue_size = 10)
        print '\033[32m' + "Converter Initialized." + '\033[0m'
    def callback(self, data):
        # Count
        self.count = self.count + 1
        print 'Count:', self.count

        # Convert
        # ipdb.set_trace()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Show
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(1)

        # Save
        '''
        time_stamp = time.asctime()# time.asctime() time.localtime() time.time()
        frames_save_path = os.path.join(Global_Params.RT_frames_save_path, time_stamp[-4:] + time_stamp[3:11] + time_stamp[11:-4].replace(':','-') + str(self.count) + '.jpg')
        cv2.imwrite(frames_save_path, cv_image)
        # '''

        # Processing
        # '''
        # ipdb.set_trace()
        frames_ARRAY[:, :, :, 0 : num_frames - 1] = frames_ARRAY[:, :, :, 1: num_frames]
        frames_ARRAY[:, :, :, num_frames - 1] = cv_image # record the latest frame
        if self.count >= num_frames:
            # ipdb.set_trace()
            get_features_from_continuous_frames(frames_ARRAY) # get features
        # '''

        # Publish
        '''
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)
        # '''

## MAIN
def main():
    ic = image_converter()
    rospy.init_node('image_converter', anonymous = True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down.")
    cv2.destroyAllWindows()

##########################
if __name__ == '__main__':
    sys.exit(main())