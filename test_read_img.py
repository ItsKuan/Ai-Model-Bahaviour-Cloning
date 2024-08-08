#!/usr/bin/env python3
###!/home/bulldog05/ai_venv/bin/python3

import os
import time, datetime
import traceback
import matplotlib.pyplot as plt
import cv2
import numpy as np
import rospy
import numpy as np
import math 
from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

MAP_SIZE = 160

class RosTopic:
    def __init__(self, save_dir):
        # self.model = model
        self.map_bridge = CvBridge()
        self.map_flag = False
        self.map_time = 0

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.folder_name = time.strftime("%Y%m%d_%H_%M_%S")
        self.save_dir = os.path.join(save_dir, self.folder_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)        

    def routed_map_callback(self, msg):
        try:
            # self.routed_map = self.bridge.compressed_imgmsg_to_cv2(msg)
            # self.routed_map = self.map_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            if msg.encoding == '8UC3':
                # print('Done getting map')
                self.routed_map = np.asarray(np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3))
            elif msg.encoding == '8UC1':
                # print('Done getting map')
                self.routed_map = np.asarray(np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width))
            elif msg.encoding == '8UC4':
                # print('Done getting map')
                self.routed_map = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                self.routed_map = cv2.cvtColor(self.route_name, cv2.COLOR_RGBA2BGR)
            else:
                rospy.logerr(f"Unsupported encoding: {msg.encoding}")
                return

            self.map_flag = True
            self.map_time = time.time()

        except Exception as e:
            rospy.logerr(f"Error converting ROS Image to OpenCV Image: {e}")

def convert_map_color(input_map):
    new_map = input_map.copy()
    L_limit=np.array([0, 0, 0]) # setting the lower limit 
    U_limit=np.array([70, 70, 70]) # setting the upper limit 
    black_mask = cv2.inRange(new_map, L_limit, U_limit)
    # print((black_mask > 0).sum())
    # print((black_mask == 0).sum())

    L_limit=np.array([150, 150, 150]) # setting the lower limit 
    U_limit=np.array([255, 255, 255]) # setting the upper limit 
    white_mask = cv2.inRange(new_map, L_limit, U_limit)
    # print((white_mask > 0).sum())
    # print((white_mask == 0).sum())
    
    new_map[black_mask > 0] = [255, 255, 255]
    new_map[white_mask > 0] = [0, 0, 0]

    return new_map

def main() :
    input_topic = RosTopic('/home/bulldog05/data/test_image')
    rospy.init_node('test_read_image')
    rospy.Subscriber("/routed_map", Image, input_topic.routed_map_callback)
   
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
    # For each new grab, mesh data is updated
        if not input_topic.map_flag:
            time.sleep(0.5)    
            continue

        input_topic.map_flag = False    
        
        now = datetime.datetime.now()
        ms = '{:03d}'.format((int)(now.microsecond/1000))

        rospy.loginfo("Saving data")      
        
        input_map = input_topic.routed_map.copy()
        fullNameImage = time.strftime("%Y_%m_%d_%H_%M_%S_") + ms + '_old.jpg'
        fullpathImage = os.path.join(input_topic.save_dir, fullNameImage)
        cv2.imwrite(fullpathImage, input_map)

        new_map = np.full(input_map.shape, [0, 0, 255])

        L_limit=np.array([0, 0, 0]) # setting the lower limit 
        U_limit=np.array([70, 70, 70]) # setting the upper limit 
        black_mask = cv2.inRange(input_map, L_limit, U_limit)
        print((black_mask > 0).sum())
        print((black_mask == 0).sum())

        L_limit=np.array([150, 150, 150]) # setting the lower limit 
        U_limit=np.array([255, 255, 255]) # setting the upper limit 
        white_mask = cv2.inRange(input_map, L_limit, U_limit)
        print((white_mask > 0).sum())
        print((white_mask == 0).sum())

        # color_pixels = np.column_stack(np.where(mask > 0))
        new_map[black_mask > 0] = [255, 255, 255]
        new_map[white_mask > 0] = [0, 0, 0]

        fullNameImage = time.strftime("%Y_%m_%d_%H_%M_%S_") + ms + '.jpg'
        fullpathImage = os.path.join(input_topic.save_dir, fullNameImage)
        cv2.imwrite(fullpathImage, new_map)

        time.sleep(1) 
           
            # print(f'Cut Image size: {image.shape}')

            # image = image[:,:,0:3]

            # cv2.imwrite('test.jpg',image)

            # Convert to PIL image

            # image = inference_transforms(image)
        rate.sleep()
          

if __name__ == "__main__" :
    main()
