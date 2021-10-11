#!/usr/bin/env python
import os
import sys
import time
import warnings
from collections import defaultdict

import cv2
import time

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
#from tqdm import tqdm
import glob
import datetime

# from utils.evalDataset import evalDataset
from .utils.mobile_model import create_mobilenetv1_ssd
from .utils.evalTransform import EvalAugmentation, AfterAugmentation
# from layers.functions.detection import *
from .utils.prior_box import *
from .utils.predictor import Predictor
# from layers.modules.multibox_loss import MultiBoxLoss
warnings.filterwarnings("ignore")

from .config import mb1_cfg, MEANS, SIZE

class Object_Detection(Node):
    def __init__(self):
        super().__init__("panorama_detection_imp")

        self.net = create_mobilenetv1_ssd(2, is_test=True)
        self.net.eval()
        self.net.to(torch.device("cuda:0"))

        net_weights = torch.load("/home/itolab-chotaro/HDD/Python/ROS2_omnidirectional_ssd_marge/src/mb1ssd_detection/mb1ssd_detection/weight/mb1-ssd-complete3.pth",
                                map_location={'cuda:0': 'cpu'})

        self.net.load_state_dict(net_weights)

        #print("into Object detection!!!!")
        start = time.time()

        self.transform = EvalAugmentation()
        self.after_transform = AfterAugmentation()

        self.Predictor = Predictor(candidate_size=200)

        self._image_pub = self.create_publisher(Image, 'person_box_image', 1)
        self._coord_pub = self.create_publisher(Float32MultiArray, 'person_box_coord', 1)
        #print("image_pablish !!!!")
        self._image_sub = self.create_subscription(Image, 'panoramaImage', self.publish_process, 1)
        #print("image_subscriber !!!")
        self._bridge = CvBridge()
        self._box_msg = Float32MultiArray()
        #print("CvBridge!!!!!")
        init_time = time.time() - start
        #print("def __init__ is ", init_time)

    def publish_process(self, data):
        cv_img = self.bridge = self._bridge.imgmsg_to_cv2(data, 'bgr8')
        complete_img, bbox = self.detection_process(cv_img)
        # リストを整形
        # bbox_sort = [box for box in bbox if len(bbox)!=0]
        bbox_sort = []
        for box in bbox:
            if len(box) != 0:
                for bx in box:
                    bbox_sort.append(bx)
        bbox_dim1 = []
        for box in bbox_sort:
            for bx in box:
                bbox_dim1.append(bx)

        print("bbox_sort : ", bbox_sort)
        if bbox_sort:
            # msgに格納
            self._box_msg.data = bbox_dim1
            dim0 = MultiArrayDimension()
            dim0.label = "foo"
            dim0.size = len(bbox_sort)
            dim0.stride = len(self._box_msg.data)
            dim1 = MultiArrayDimension()
            dim1.label = "bar"
            dim1.size = len(bbox_sort[0])
            dim1.stride = dim1.size

            self._box_msg.layout.dim = [dim0, dim1]
        
            self._coord_pub.publish(self._box_msg)

        # print("complete_img : ", complete_img.shape)
        self._image_pub.publish(self._bridge.cv2_to_imgmsg(complete_img, "bgr8"))
        pub_end = time.time()

    def detection(self, img, only_front=True):
        boxes_list = []
        score_list = []
        img_ori = img
        img, _, _ = self.transform(img, "", "")
        print("img is", img.size())
        img = img.to(torch.device("cuda:0"))
        scores, boxes = self.net(img[1:3])
        img = img.to(torch.device("cpu"))
        # img, _, _ = self.after_transform(img, "", "")
        # print("img is ", img.shape)
        # boxes, labels, score = self.Predictor.predict(score, boxes, 10, 0.4)
        #print("boxes is", boxes.size())
        for box, score in zip(boxes, scores):
            # box = box.unsqueeze(0)
            # score = score.unsqueeze(0)
            box, labels, score = self.Predictor.predict(score, box, 100, 0.4)
            # img = img[:, :, [0, 1, 2]]
            # print("box :", box.squeeze())
            boxes_list.append(box.tolist())
            score_list.append(score.tolist())


        return img_ori, score_list, boxes_list

    def get_coord(self, img, boxes_list, only_front):
        #print(prediction_box)
        #cv2.namedWindow("Image")
        #cv2.imshow("image", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #print(np.shape(img))

        img = np.uint8(img)

        img = cv2.resize(img, (1200, 300)) #よくわからないけど, これがないとエラー
        img = cv2.line(img, (300, 0), (300, 300), (0, 255, 0), 2)
        img = cv2.line(img, (900, 0), (900, 300), (0, 255, 0), 2)
        height, width, _ = img.shape
        print(type(img))
        # print("img is ", img)
        if only_front == True:
            print("width :", width)
            print("boxes_list : ", boxes_list)
            for i, boxes in enumerate(boxes_list):
                print("boxes is ", boxes)
                if boxes and i == 0:
                    for p, box in enumerate(boxes):
                        img = cv2.rectangle(img, (np.int(box[0] + 1 * width/4), np.int(box[1])), (np.int(box[2] + 1 * width/4), np.int(box[3])), (255, 0, 0), 5)
                        boxes_list[i][p][0] += 1*width/4
                        boxes_list[i][p][2] += 1*width/4
                elif boxes and i == 1:
                    for p, box in enumerate(boxes):
                        img = cv2.rectangle(img, (np.int(box[0] + 2 * width/4), np.int(box[1])), (np.int(box[2] + 2 * width/4), np.int(box[3])), (255, 0, 0), 5)
                        boxes_list[i][p][0] += 2*width/4
                        boxes_list[i][p][2] += 2*width/4
            print("boxes_list after : ", boxes_list)

        else:
            for i, boxes in enumerate(boxes_list):
                    img = cv2.rectangle(img, (np.int(boxes[0] + i * width/4), np.int(boxes[1])), (np.int(boxes[2] + i * width/4), np.int(boxes[3])), (255, 0, 0), 5)
        return img, boxes_list



    def detection_process(self, img):
        only_front = True
        img, score, prediction_bbox = self.detection(img)
        # print("score is ", score.size())
        # print("predictbox is ", prediction_bbox.size())
        img, prediction_bbox = self.get_coord(img, prediction_bbox, only_front=only_front)
        print("img : ", img.shape)
        return img, prediction_bbox
