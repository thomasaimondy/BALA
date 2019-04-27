#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import ipdb
import cv2
import ipdb



csv_data = pd.read_csv('connections.csv', sep=',')

csv_data = csv_data[csv_data['Weight'] > 30]

df = pd.DataFrame(csv_data)

df.to_csv('connections_filt_30.csv')

# csv_data = csv_data[csv_data['Language'] == 'English']





# sentences = csv_data['Description']
