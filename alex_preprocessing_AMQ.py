#-*- coding: utf-8 -*-
import sys
import time
import cv2
import numpy as np
from cnn_util import CNN
import stomp
import skimage
import Global_Params
import ipdb

########################### Global Parameters ###############################
num_frames = Global_Params.num_frames #40
########################### Global Parameters ###############################

############################### Initialization ###################################
# CNN model
vgg_model = '/home/thomas/caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
vgg_deploy = '/home/thomas/caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'
cnn = CNN(model = vgg_model, deploy = vgg_deploy, width = 224, height = 224)

# Time-Series ndArray
frames_ARRAY = np.zeros((Global_Params.img_height, Global_Params.img_width, 3, num_frames), dtype = np.uint8)# 4-dims
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
    for t in xrange(0, num_frames):# acess single frame one by one
        frame_list.append(frames_array[:, :, :, t])

    # ipdb.set_trace()
    frame_list = np.array(frame_list)
    cropped_frame_list = np.array(map(lambda x: preprocess_frame(x), frame_list))# preprocess each frame

    feats = cnn.get_features(cropped_frame_list)# get continuous features
    np.save(Global_Params.RT_feats_save_fullpath, feats)
    print 'Features saved.'

## Connection Server
class MyListener(object):
    def __init__(self):
        self.count = 0
    def on_error(self, headers, message):
        # print 'received an error: %s' % message
        print 'Received an ERROR'
    def on_message(self, headers, message):
        # print 'received a message: %s' % message
        self.count = self.count + 1
        print 'Count:', self.count, 'Length:', len(message)# type(message) = str

        # ipdb.set_trace()
        img_array = np.fromstring(message, dtype = np.uint8)
        img = cv2.imdecode(img_array, cv2.CV_LOAD_IMAGE_COLOR)
        # cv2.imshow('IMG', img)
        # cv2.waitKey(1)
        # cv2.imwrite('/home/thomas/Desktop/img.jpg', img)

        # ipdb.set_trace()
        frames_ARRAY[:, :, :, 0 : num_frames - 1] = frames_ARRAY[:, :, :, 1: num_frames]
        frames_ARRAY[:, :, :, num_frames - 1] = img# record the latest frame

        if self.count >= num_frames:
            # ipdb.set_trace()
            get_features_from_continuous_frames(frames_ARRAY)# First: get features

## MAIN
def main():
    # INITIAL connection
    conn = stomp.Connection([('192.168.0.103', 61613)])
    conn.set_listener('MyListener', MyListener())
    conn.start()
    conn.connect(wait = True, headers = {'client_id': 'beauty', 'non_persistent': 'true'})

    # RECIEVE messages
    conn.subscribe(destination = '/topic/Picture.FOO', id = 1, ack = 'auto', headers = {'activemq.subscriptionName':'beauty'})
    time.sleep(1)

    # SEND messages
    # conn.send(body = 'hello,BEAUTY!', destination = 'Picture.FOO')
    # y = 'hello,garfield! this is '.join(sys.argv[1:]), destination = '/queue/test')
    # time.sleep(1)

    while(True):
        pass
    conn.disconnect()

##########################
if __name__ == '__main__':
    sys.exit(main())