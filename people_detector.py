#!/usr/bin/python
import argparse
import tools.find_mxnet
import mxnet as mx
import os
import sys
from detect.detector import Detector
from symbol.symbol_factory import get_symbol
import cv2

import mysql.connector
import datetime
import time



MYSQL_HOST = ""
MYSQL_USER = ""
MYSQL_PASS = ""
MYSQL_DB   = ""

NETWORK = 'resnet50'
EPOCH = 0
PREFIX = os.path.join(os.getcwd(), 'model', 'ssd_')
CPU = True
DATA_SHAPE = 512
MEAN_R = 123.0
MEAN_G = 117.0
MEAN_B = 104.0
THRESH = 0.5
NMS_THRESH = 0.5
FORCE_NMS = True
TIMER=False
DEPLOY_NET = False
#CLASS_NAMES = 'person, tvmonitor'
CLASS_NAMES = 'aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor'
IMAGE_WIDTH=1024
IMAGE_HEIGHT=768


class CamHandler:
        def __init__(self):
                DEVICE = '/dev/video0'
                self.SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
                self.camera = cv2.VideoCapture(0)

        def get_camera_image(self):
                (ret,img) = self.camera.read()
                return img

	def save_image(self,image,name):
		cv2.imwrite(name,image)

def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx, num_class,
                 nms_thresh=0.5, force_nms=True, nms_topk=400):
    if net is not None:
        net = get_symbol(net, data_shape, num_classes=num_class, nms_thresh=nms_thresh,
            force_nms=force_nms, nms_topk=nms_topk)
    detector = Detector(net, prefix, epoch, data_shape, mean_pixels, ctx=ctx)
    return detector


def parse_class_names(class_names):
    """ parse # classes and class_names if applicable """
    if len(class_names) > 0:
        if os.path.isfile(class_names):
            # try to open it to read class names
            with open(class_names, 'r') as f:
                class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in class_names.split(',')]
        for name in class_names:
            assert len(name) > 0
    else:
        raise RuntimeError("No valid class_name provided...")
    return class_names

if __name__ == '__main__':

    cam = CamHandler()

    img = cam.get_camera_image()
    cam.save_image(img,'/tmp/011.jpg') 
    ctx = mx.cpu()


    image_list = ['/tmp/011.jpg']


    network = None if DEPLOY_NET else NETWORK
    network = NETWORK
    class_names = parse_class_names(CLASS_NAMES)
    prefix = PREFIX
    if prefix.endswith('_'):
    	prefix = PREFIX + NETWORK + '_' + str(DATA_SHAPE)
    detector = get_detector(network, prefix, EPOCH,
                            DATA_SHAPE,
                            (MEAN_R, MEAN_G, MEAN_B),
                            ctx, len(class_names), NMS_THRESH, FORCE_NMS)
    # run detection
    result = detector.detect_and_visualize(image_list, None, None,
                                  class_names, THRESH, TIMER)
    counter = 0

    for r in result[0]:
	if r[0]=='person':
		counter+=1
    print ("People detected %i" % counter)
    conn = mysql.connector.connect(host=MYSQL_HOST,database=MYSQL_DB,user=MYSQL_USER,password=MYSQL_PASS)
    currentDate = time.strftime('%Y-%m-%d %H:%M:%S')
    query = "INSERT INTO peoplecounter(timer,counter) VALUES (%s,%s)"
    cursor = conn.cursor()
    print [currentDate,counter]
    cursor.execute(query,[currentDate,str(counter)])

    conn.commit()
    conn.close()
