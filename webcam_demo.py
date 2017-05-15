# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:01:42 2017

@author: zck
"""


import os, time
import math
import random

import numpy as np
import cv2
from facenet.src.align import detect_face
import tensorflow as tf

if __name__ == '__main__':
    gpu_memory_fraction = 0.25
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, 'facenet/src/align')
            
    cap = cv2.VideoCapture(0)
    while True:
        if not cap.isOpened():
            print('Failed to load camera.')
            time.sleep(5)
            pass
        # Capture frame-by-frame
        ret, frame = cap.read()        
#        print(ret)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        img = np.copy(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        bounding_boxes, landmarks = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        
        for bb in np.array(bounding_boxes, np.int32):
            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255,0,0), 2)
        cv2.imshow('demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Save Recently captured frame as Jpeg file 
#        cv2.imwrite(path + 'captured.jpg', frame)
#        img = mpimg.imread(path + 'captured.jpg')
        
#        rclasses, rscores, rbboxes =  process_image(img)
    
        # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
#        visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
    
        
    cap.release()
    cv2.destroyAllWindows()