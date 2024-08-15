#!/usr/bin/env python3
###!/home/bulldog05/ai_venv/bin/python3

import math
import os
import time
from collections import deque
from pathlib import Path

# import traceback
# import matplotlib.pyplot as plt
import cv2
import numpy as np
import onnxruntime as ort
import rospy
import torch
from cargobot_msgs.msg import (DriveState, GlobalPath, LaneFollowing,
                               MapStatus, MotionStatus, RobotStatus, Safety)
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
# from collect_data.msg import RemoteControl
from sensor_msgs.msg import CompressedImage, Image, Imu, NavSatFix
from std_msgs.msg import String

# ROOT_DIR = Path(os.path.abspath(__file__)).parents[1]
MODEL_FOLDER = "models"
# MODEL_PATH = '/home/bulldog01/catkin_ws/src/dev/behaviour_cloning/models/bulldog_sumtn_v10_256_e29_a5000.onnx'
# MODEL_PATH = '/home/bulldog01/catkin_ws/src/dev/behaviour_cloning/models/bulldog_sumtn_v11_256_e28_a5000.onnx' 

# MODEL_PATH = '/home/bulldog01/catkin_ws/src/dev/behaviour_cloning/models/bulldog_sumtn_v11_256_e86_a5000.onnx' # best 1st batch 

#MODEL_PATH = '/home/bulldog01/catkin_ws/src/dev/behaviour_cloning/models/bulldog_sumtn_v15_256_e25_a5000.onnx' # best for demo
MODEL_PATH = '/home/asimovsimpc/model_weight/bulldog_sumtn_v15_256_e74_a5000.onnx' #TEST 2
# MODEL_PATH = '/home/bulldog01/catkin_ws/src/dev/behaviour_cloning/models/bulldog_sumtn_v15_256_e53_a5000.onnx'

# MODEL_PATH = '/home/bulldog01/catkin_ws/src/dev/behaviour_cloning/models/bulldog_sumtn_v15_256_e74_a5000.onnx'
# MODEL_PATH = '/home/bulldog01/catkin_ws/src/dev/behaviour_cloning/models/bulldog_sumtn_v16_256_e55_a100.onnx'

# RESIZE_DIM = (320, 240)
RESIZE_DIM = (256, 256)
AI_RATE = 45

def center_crop(img, dim):
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped from center
    """
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 

    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

def normalize_cv2(img, mean, denominator):
    if mean.shape and len(mean) != 4 and mean.shape != img.shape:
        mean = np.array(mean.tolist() + [0] * (4 - len(mean)), dtype=np.float64)
    if not denominator.shape:
        denominator = np.array([denominator.tolist()] * 4, dtype=np.float64)
    elif len(denominator) != 4 and denominator.shape != img.shape:
        denominator = np.array(denominator.tolist() + [1] * (4 - len(denominator)), dtype=np.float64)

    img = np.ascontiguousarray(img.astype("float32"))
    cv2.subtract(img, mean.astype(np.float64), img)
    cv2.multiply(img, denominator.astype(np.float64), img)
    return img

def normalize_numpy(img, mean, denominator):
    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img

def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    if img.ndim == 3 and img.shape[-1] == 3:
        return normalize_cv2(img, mean, denominator)
    return normalize_numpy(img, mean, denominator)

def convert_rad_to_degree(rad):
    rad = (rad + np.pi) % (2 * np.pi)
    deg = math.degrees(rad)
    deg = max(0, min(deg, 360))
    return np.int64(deg)

class SteeringPredictor:
    # Create Deep learning object to predict steering angular from frame
    def __init__(self, camera_name):
        # self.model = model
        self.lane_follow_ready = 1
        self.drive_mode = 2  # Drive mode:   0~RC  1~Tele  2~AI
        self.camera_name = camera_name
        self.image_bridge = CvBridge()
        self.map_bridge = CvBridge()
        self.image = None  # (np.random.rand(IMG_HEIGHT, IMG_WIDTH, 3) * 255).astype(np.uint8)
        self.imu = None  # np.random.randn(1, 4).astype(np.float32)
        self.global_path = None  # np.random.randn(1, 1, 128, 128).astype(np.float32)
        self.yaw = None  # np.random.randn(1, 1).astype(np.float32)
        self.gps_error = None  # np.random.randn(1, 1).astype(np.float32)
        self.routed_map = None
        self.time = 0
        self.linear_x = 0
        self.angular_z = 0
        self.old_time = 0
        self.old_seq = 0
        self.seq = 0
        self.image_flag = False
        self.map_flag = False
        self.map_time = 0
        self.road_type = 0
        self.road_width = 0
        self.high_cmd = 0
        self.front_limit_speed = 1
        self.rear_limit_speed = -1
        self.contermet = 0

    def drive_state_callback(self, data):
        # Get Drive mode state from Drive Mode node
        drive_state = data
        try:
            self.drive_mode = drive_state.drive_mode_state
        except Exception as e:
            print(e)

    def odom_callback(self,data):
        self.linear_x = data.twist.twist.linear.x
        self.angular_z = data.twist.twist.linear.z

    def motion_callback(self,data):
        self.front_limit_speed = data.threshold_linear_above
        self.rear_limit_speed = data.threshold_linear_below

    def robot_status_callback(self,data):
        self.contermet = data.contermet_all
    
    def gps_error_callback(self, data):
        # Get Drive mode state from Drive Mode node
        try:
            gps_error = (data.position_covariance[0] + data.position_covariance[4]) * 0.5
            self.gps_error = gps_error
        except Exception as e:
            print(e)

    def image_callback(self, msg):
        try:
            self.image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            secs = msg.header.stamp.secs
            nsecs = msg.header.stamp.nsecs
            self.old_time = self.time
            self.time = secs + nsecs*1e-9
            self.old_seq = self.seq
            self.seq = msg.header.seq

            # print(f"the waiting time between 2 consecutive frames is {self.time - self.old_time} seconds")
            # print(f"Seq step size {self.seq - self.old_seq} seconds")

        except Exception as e:
            print(e)

    def compressed_image_callback(self, msg):
        try:
            self.image = self.image_bridge.compressed_imgmsg_to_cv2(msg)
            secs = msg.header.stamp.secs
            nsecs = msg.header.stamp.nsecs
            self.old_time = self.time
            self.time = secs + nsecs*1e-9
            self.old_seq = self.seq
            self.seq = msg.header.seq
            self.image_flag =True
            # print(f"the time between 2 consecutive frames is {self.time - self.old_time} seconds")
            # print(f"Seq step size {self.seq - self.old_seq} frames")


        except Exception as e:
            print(e)

    def routed_map_callback(self, msg):
        try:
            # self.routed_map = self.bridge.compressed_imgmsg_to_cv2(msg)
            # self.routed_map = self.map_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            if msg.encoding == '8UC3':
                # print('Done getting map')
                self.routed_map = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding == '8UC1':
                # print('Done getting map')
                self.routed_map = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            elif msg.encoding == '8UC4':
                # print('Done getting map')
                self.routed_map = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                self.routed_map = cv2.cvtColor(self.routed_map, cv2.COLOR_RGBA2BGR)
            else:
                rospy.logerr(f"Unsupported encoding: {msg.encoding}")
                return

            self.map_flag = True
            self.map_time = time.time()

        except Exception as e:
            rospy.logerr(f"Error converting ROS Image to OpenCV Image: {e}")

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        
        return roll_x, pitch_y, yaw_z # in radians

    def status_callback(self, data):
        try:            
            self.road_type = data.service
            self.road_width = data.width_of_path
            self.high_cmd = data.high_cmd


        except Exception as e:
            print(e)
            pass

    def imu_callback(self, data):
        try:
            roll, pitch, yaw = self.euler_from_quaternion(
                data.orientation.x,
                data.orientation.y,
                data.orientation.z,
                data.orientation.w
            )
            roll, pitch, yaw = np.float32(roll), np.float32(pitch), np.float32(yaw)
            self.roll, self.pitch, self.yaw = roll, pitch, yaw

        except Exception as e:
            print(e)
            pass

    def process_grid_map(self, global_path):

        def get_global_path(gp):
            gp = np.array(gp, dtype=np.float32)
            gp = gp[11:] # trigger
            return gp

        proc_global_path = get_global_path(global_path)

        return proc_global_path

    def global_path_callback(self, data):
        try:
            
            self.global_path = self.process_grid_map(eval(data.array_path))

        except Exception as e:
            print(e)

    def remote_control_callback(self, data):
        try:
            lv = data.remote_control.remote_vel_cmd.linear.x
            av = data.remote_control.remote_vel_cmd.angular.z
            self.lv = np.float32(lv)
            self.av = np.float32(av)
        except Exception as e:
            print(e)

    @staticmethod
    def finetune(angular_velocity, linear_velocity, ratio):
        if ratio > 1.5:
            ratio = 1.5
        if abs(angular_velocity) > 0.04 and abs(angular_velocity) <= 0.10:
            angular_velocity = angular_velocity * (ratio * 0.8)
        elif abs(angular_velocity) > 0.10 and abs(angular_velocity) <= 0.20:
            angular_velocity = angular_velocity * (ratio * 0.85)
        elif abs(angular_velocity) > 0.20 and abs(angular_velocity) <= 0.30:
            angular_velocity = angular_velocity * (ratio * 0.9)
        elif abs(angular_velocity) > 0.30 and abs(angular_velocity) <= 0.40:
            angular_velocity = angular_velocity * (ratio * 0.95)
        elif abs(angular_velocity) > 0.40:
            angular_velocity = angular_velocity * ratio

        linear_velocity = linear_velocity * ratio

        return linear_velocity, angular_velocity

def convert_map_color(input_map):
    # print(input_map.shape)
    new_map = np.full(input_map.shape, [255, 0, 0])
    L_limit=np.array([0, 0, 0]) # setting the lower limit 
    U_limit=np.array([70, 70, 70]) # setting the upper limit 
    black_mask = cv2.inRange(input_map, L_limit, U_limit)
    # print((black_mask > 0).sum())
    # print((black_mask == 0).sum())

    L_limit=np.array([150, 150, 150]) # setting the lower limit 
    U_limit=np.array([255, 255, 255]) # setting the upper limit 
    white_mask = cv2.inRange(input_map, L_limit, U_limit)
    # print((white_mask > 0).sum())
    # print((white_mask == 0).sum())
    
    new_map[black_mask > 0] = [255, 255, 255]
    new_map[white_mask > 0] = [0, 0, 0]

    return new_map

def main():

    steering_predictor = SteeringPredictor('front')
    rospy.init_node('fastvit_temporal')#, anonymous=True)

    rospy.Subscriber("/drive_state", DriveState, steering_predictor.drive_state_callback)
    # rospy.Subscriber("/zed2i/zed_node/left_raw/image_raw_color", Image, steering_predictor.image_callback) # origin
    rospy.Subscriber("/camera_front/camera/image_raw/compressed", CompressedImage, steering_predictor.compressed_image_callback)
    # rospy.Subscriber("/cameav_ra_front/image_rect_color/compressed", CompressedImage, steering_predictor.compressed_image_callback)
    # rospy.Subscriber("/routed_map/compressed", CompressedImage, steering_predictor.routed_map_callback)
    rospy.Subscriber("/routed_map", Image, steering_predictor.routed_map_callback)
    rospy.Subscriber("/map_status", MapStatus, steering_predictor.status_callback)
    # rospy.Subscriber("/imu/data", Imu, steering_predictor.imu_callback)
    # rospy.Subscriber("/global_path_nearest", GlobalPath, steering_predictor.global_path_callback)
    # rospy.Subscriber("/ublox/fix", NavSatFix, steering_predictor.gps_error_callback)
    # rospy.Subscriber("/remote_control", RemoteControl, steering_predictor.remote_control_callback)
    rospy.Subscriber("/odom", Odometry, steering_predictor.odom_callback)
    rospy.Subscriber("/motion_stt", MotionStatus, steering_predictor.motion_callback)
    rospy.Subscriber("/robot_status", RobotStatus, steering_predictor.robot_status_callback)

    lane_follow_msg = LaneFollowing()
    pub_lane_follow_cmd = rospy.Publisher('/lane_follow_cmd', LaneFollowing, queue_size=1)
    # pub_trajs = rospy.Publisher('/trajs', String, queue_size=1)

    # Warning: onnxruntime still is not optimized (parameters are not optimized)
    # write all config tensorrtexecutionprovider
    ort_option = {'device_id': 0,
                  'trt_max_workspace_size': 3221225472, #2147483648*2
                  'trt_engine_cache_enable': True,
                  'trt_fp16_enable': True
                  }
    
    ort_session = ort.InferenceSession(MODEL_PATH, providers=[
        #('TensorrtExecutionProvider', ort_option),
        #('CUDAExecutionProvider',ort_option),
        'CUDAExecutionProvider','CPUExecutionProvider'
        ])

    # print providers
    print('---'*5)
    print("Available providers:") 
    for provider in ort_session.get_providers():
        print(provider)
    print('---'*5)

    print(ort_session.get_inputs())

    if 'bc' in MODEL_PATH.split('/')[-1]:        
        model_name = "bc"
    else:
        model_name = "umtn"
    lane_follow_msg.model_name = model_name
    
    input_name = ort_session.get_inputs()[0].name
    route_name = ort_session.get_inputs()[1].name
    # feature_name = ort_session.get_inputs()[2].name
    # last_action_name = ort_session.get_inputs()[3].name

    # image = np.random.rand(1, 3, 256, 256).astype(np.float32)
    # routed_map = np.random.rand(1, 3, 128, 128).astype(np.float32)
    # for i in range(100):
    #     start_time = time.time()
    #     ort_inputs = {input_name: image, route_name: routed_map}#, feature_name: features_buffer, last_action_name: last_action}
    #     pred_lv, pred_av, pred_prob = ort_session.run(None, ort_inputs)
    #     best_mode = np.argmax(pred_prob[0])
    #     best_pred_lv = pred_lv[0][best_mode]
    #     best_pred_av = pred_av[0][best_mode]
    #     fps = 1 / (time.time() - start_time)
    #     lv = pred_lv[0][0]
    #     av = pred_av[0][0]
    #     # lv1, av1 = steering_predictor.finetune(av, lv, 1.25)
    #     print(f'Output shape: {pred_lv.shape}:{pred_av.shape} {best_pred_av.shape}, fps: {fps:.2f}')
    
    # exit()

    rate = rospy.Rate(AI_RATE) # change 30 to 60
    # # Create a mesh object
    count = 0
    crr_time = -1
    count_dupplicate = 0
    start_idx = True

    old_time = -1
    start_dupplicate = time.time()

    is_stucked = False
    try_to_escape = False
    start_to_turn_left = False

    begin_limit_time = time.time()
    begin_contermet = -1
    begin_contermet_escape = -1
    begin_contermet_left_turn = -1

    move_back_distance = 15e-4 # km
    stuck_wait_time = 2 # seconds
    
    back_speed = -0.8
    back_angular = -0.1
    escape_speed = 1.5
    travel_distance = 0
    escape_angular = 0.05

    # features_buffer = np.zeros((1, 3, 512), dtype=np.float32)
    # last_action = np.zeros((1, 2), dtype=np.float32)
    # queue = deque(maxlen=2)

    while not rospy.is_shutdown():
        # stop AI if in RC mode (0) or joystick mode (3)
        if steering_predictor.drive_mode == 0 or steering_predictor.drive_mode == 3: 
            print("----------------- Not in AI mode ---------")
            time.sleep(1)
            continue        

        # get image from sub
        image = steering_predictor.image
        routed_map = steering_predictor.routed_map
        
        # image = np.random.randint(255, size=(RESIZE_DIM[0], RESIZE_DIM[1], 3),dtype=np.uint8)
        # routed_map = np.random.randint(255, size=(160, 160, 3),dtype=np.uint8)

        if not steering_predictor.image_flag:
            # print(f"--------------------------------- Waiting for image here!")
            time.sleep(0.01)    
            continue

        if not steering_predictor.map_flag:
            if (time.time() - steering_predictor.map_time) > 1:
                print(f"---------------------------- Waiting for routed map here!------------------")
                time.sleep(1)    
                continue

        # check if robot is stucked by obstacle ahead
        if steering_predictor.front_limit_speed <= 0.5:
            if not is_stucked:
                if time.time() - begin_limit_time > stuck_wait_time:
                    is_stucked = True
        else:
            begin_limit_time = time.time()
        
        # try to move back when robot got stucked
        if is_stucked and steering_predictor.front_limit_speed < 1.:
            if begin_contermet == -1:
                begin_contermet = steering_predictor.contermet

            travel_distance = steering_predictor.contermet - begin_contermet
            if travel_distance < move_back_distance:
                lane_follow_msg.lane_follow_ready = 1
                lane_follow_msg.lane_follow_vel.linear.x = back_speed * max((move_back_distance - travel_distance)/move_back_distance, 0.3)
                lane_follow_msg.lane_follow_vel.angular.z += back_angular
                pub_lane_follow_cmd.publish(lane_follow_msg)
                print("--------- Robot is moving backward -----")
                time.sleep(0.02)
                continue
            else:
                if is_stucked:
                    try_to_escape = True
                    begin_contermet_escape = steering_predictor.contermet

                is_stucked = False
                begin_contermet = -1
        else:
            if is_stucked:
                try_to_escape = True
                begin_contermet_escape = steering_predictor.contermet

            is_stucked = False
            begin_contermet = -1

        lane_follow_msg.stuck_camera = 0
        
        # if old_time != steering_predictor.time:
        #     pass
        #     start_dupplicate = time.time()
        #     old_time = steering_predictor.time                

        # else:
        #     pass
        #     # print(f"the waiting time between 2 consecutive frames is {time.time() - start_dupplicate} seconds ----------------")
        #     time.sleep(0.01)
        #     continue
        
        start_time = time.time()

        H, W, _ = image.shape
        # image = image[H//3:,:]
        image = cv2.resize(image, (256,256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        image = normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0).astype(np.float32)

        # add image to queue
        # queue.append(image)

        # if len(queue) != 2:
        #     continue

        # concat_image = np.concatenate([queue[0], queue[1]], axis=1)

        routed_map = routed_map[80-64:80+64, 80-64:80+64]
        
        # routed_map = new_map
        # routed_map = cv2.resize(routed_map, (64, 64))
        routed_map = cv2.cvtColor(routed_map, cv2.COLOR_BGR2RGB)
        
        # convert new map to old map color
        # routed_map = convert_map_color(routed_map)
        
        routed_map = routed_map / 255.0
        routed_map = np.transpose(routed_map, [2,0,1]).astype(np.float32)
        routed_map = np.expand_dims(routed_map, axis=0).astype(np.float32)

        # print(f'Last action: {last_action}')

        ort_inputs = {input_name: image, route_name: routed_map}#, feature_name: features_buffer}
        pred_lv, pred_av = ort_session.run(None, ort_inputs)
        # pred_lv, pred_av, pred_prob = ort_session.run(None, ort_inputs)

        # last_action = pred_action #ort_outs[0][0].reshape(1, 2) # update last action
        # feature_buffer = ort_outs[0][1].reshape(1, 512) # get feature buffer

        # features_buffer[:, :-1] = features_buffer[:, 1:]
        # features_buffer[:, -1] = feature_buffer

        # control, trajs, map_head = ort_outs
        # control = ort_outs

        # best_mode = np.argmax(pred_prob[0])
        # best_pred_lv = pred_lv[0][best_mode]
        # best_pred_av = pred_av[0][best_mode]

        # linear_vel = best_pred_lv[0] #pred_lv[0][0]
        # angular_vel = best_pred_av[0] #pred_av[0][0]

        linear_vel = pred_lv[0][0]
        angular_vel = pred_av[0][0]

        # angular_vel += 0.01 # for model v15_e53/e28
        # angular_vel += 0.01 # for model v16_e55
        
        if (steering_predictor.road_type == 3): # barrier
            print(f'------------------------------ Slowing down speed because of barrier------status = {steering_predictor.road_type}')
            linear_vel = min(linear_vel, 0.7)
            pass

        elif (steering_predictor.road_type == 2): # speed bump
            print(f'------------------------------ Slowing down speed because of speed bump------status = {steering_predictor.road_type}')
            linear_vel = min(linear_vel, 1.)
        # if abs(angular_vel) < 0.025 and abs(linear_vel) < 0.8:
        #     last_action = np.array([0.8, angular_vel]).reshape(1, 2).astype(np.float32)

        if try_to_escape:
            if (steering_predictor.contermet - begin_contermet_escape < 0.0015) and (steering_predictor.front_limit_speed < 1.5):
                print("---- try to escape here -----")
                angular_vel += escape_angular
                linear_vel = min(linear_vel, escape_speed)

            else:
                try_to_escape = False
                begin_contermet_escape = -1
        else:
            if start_to_turn_left:
                print("---- try to slow down when turning left -----")
                if steering_predictor.contermet - begin_contermet_left_turn < 0.001:
                    linear_vel = min(linear_vel, 0.5)
                else:
                    start_to_turn_left = False

            elif (steering_predictor.angular_z > 0.2):
                start_to_turn_left = True
                begin_contermet_left_turn = steering_predictor.contermet
                linear_vel = min(linear_vel, 0.5)


        lane_follow_msg.lane_follow_ready = 1
        lane_follow_msg.lane_follow_vel.linear.x = linear_vel#[0]
        lane_follow_msg.lane_follow_vel.angular.z = angular_vel#[0]
        pub_lane_follow_cmd.publish(lane_follow_msg)

        final_fps = 1 / (time.time() - start_time)
        
        # print(f'AV: {angular_vel} | LV: {linear_vel} | FPS: {final_fps:.2f} | Dup: {count_dupplicate} times')
        print(f'AV: {angular_vel} | LV: {linear_vel} | FPS: {final_fps:.2f}')

        steering_predictor.image_flag = False
        steering_predictor.map_flag = False

        # steering_predictor.routed_map = None
        # steering_predictor.image = None

        rate.sleep()

if __name__ == "__main__":
    main()
