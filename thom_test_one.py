#-*- coding: utf-8 -*-

from thom_preprocessing import get_features_from_video_filemode
from thom_model import test_file

root_path = '/home/bibo/Data/video_to_sequence/'
video_src_path = root_path + 'data/test_one_video/'
feat_save_path = root_path + 'data/test_one_feature/'
num_frames = 20
get_features_from_video_filemode(video_src_path, feat_save_path,num_frames)


model_path = root_path + 'models_new/model-2900'
video_feat_path = root_path + 'data/test_one_feature/task1/'
ixtoword_path = root_path + 'data/ixtoword.npy'
#video_feat = preprocess.get_features_from_video()
test_file(model_path,video_feat_path,ixtoword_path,num_frames)
