#!/usr/bin/env python
#-*- coding: utf-8 -*-
# OLD_CHECKPOINT_FILE = "/home/bibo/Data/video_to_sequence/models/model-900"
# NEW_CHECKPOINT_FILE = "/home/bibo/Data/video_to_sequence/models/model2"

OLD_CHECKPOINT_FILE = "/home/bibo/Data/video_to_sequence/models_alex_0/model-900"
NEW_CHECKPOINT_FILE = "/home/bibo/Data/video_to_sequence/models_alex_0/model2"


import tensorflow as tf
vars_to_rename = {
    "LSTM1/BasicLSTMCell/Linear/Matrix": "LSTM1/basic_lstm_cell/kernel",
    "LSTM2/BasicLSTMCell/Linear/Matrix": "LSTM2/basic_lstm_cell/kernel",
    "LSTM1/BasicLSTMCell/Linear/Bias": "LSTM1/basic_lstm_cell/bias",
    "LSTM2/BasicLSTMCell/Linear/Bias": "LSTM2/basic_lstm_cell/bias",
    "Wemb":"Wemb",
    "embed_word_W":"embed_word_W",
    "encode_image_b":"encode_image_b",
    "encode_image_W":"encode_image_W",
    "embed_word_b":"embed_word_b",
}
# LSTM1/basic_lstm_cell/bias
new_checkpoint_vars = {}
reader = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
for old_name in reader.get_variable_to_shape_map():
    print(old_name)
    if old_name in vars_to_rename:
        new_name = vars_to_rename[old_name]
        new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))
    # else:
    #     new_name = old_name
    #     new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

init = tf.global_variables_initializer()
saver = tf.train.Saver(new_checkpoint_vars)

with tf.Session() as sess:
    sess.run(init)
    saver.save(sess, NEW_CHECKPOINT_FILE)


    # Wemb
# LSTM2/BasicLSTMCell/Linear/Matrix
# LSTM1/BasicLSTMCell/Linear/Bias
# embed_word_W
# encode_image_b
# LSTM2/BasicLSTMCell/Linear/Bias
# encode_image_W
# embed_word_b
# LSTM1/BasicLSTMCell/Linear/Matrix
