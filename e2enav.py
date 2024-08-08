#!/usr/bin/env python3
###!/home/bulldog05/ai_venv/bin/python3

import os
import time
import traceback
import matplotlib.pyplot as plt
import cv2
import numpy as np
import onnxruntime as ort
import rospy
import torch
import numpy as np
import math 
from std_msgs.msg import String
from cargobot_msgs.msg import DriveState,Safety
from cargobot_msgs.msg import LaneFollowing
from cargobot_msgs.msg import GlobalPath
from collect_data.msg import RemoteControl
from sensor_msgs.msg import NavSatFix
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import Imu
from pathlib import Path
from nav_msgs.msg import Odometry,Path
from collections import deque
from cargobot_msgs.msg import State,StateArray
from geometry_msgs.msg import PoseStamped

MODEL_FOLDER = "models"
MODEL_PATH = '/home/bulldog01/catkin_ws/src/dev/behaviour_cloning/models/e2e_nav_temporal_n08_e119_gtx4090.onnx' 

RESIZE_DIM = (256, 256)
AI_RATE = 30


def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return qx, qy, qz, qw


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
                self.routed_map = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding == '8UC1':
                self.routed_map = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            elif msg.encoding == '8UC4':
                self.routed_map = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                self.routed_map = cv2.cvtColor(self.route_name, cv2.COLOR_RGBA2BGR)
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
    rospy.init_node('e2enav_temporal')#, anonymous=True)
    rospy.Subscriber("/drive_state", DriveState, steering_predictor.drive_state_callback)
    rospy.Subscriber("/camera_front/camera/image_raw/compressed", CompressedImage, steering_predictor.compressed_image_callback)
    rospy.Subscriber("/routed_map", Image, steering_predictor.routed_map_callback)
    rospy.Subscriber("/odom", Odometry, steering_predictor.odom_callback)
    traj_pub = rospy.Publisher("trajectory_ref",StateArray,queue_size=1,tcp_nodelay=True)
    path_pub = rospy.Publisher("path_pred",Path,queue_size=1,tcp_nodelay=True)

    lane_follow_msg = LaneFollowing()
    # pub_lane_follow_cmd = rospy.Publisher('/lane_follow_cmd', LaneFollowing, queue_size=1)

    # Warning: onnxruntime still is not optimized (parameters are not optimized)
    # write all config tensorrtexecutionprovider
    ort_option = {'device_id': 0,
                  'trt_max_workspace_size': 3221225472, #2147483648*2
                  'trt_engine_cache_enable': True,
                  'trt_fp16_enable': True
                  }
    
    ort_session = ort.InferenceSession(MODEL_PATH, providers=[
        ('TensorrtExecutionProvider', ort_option),
        'CUDAExecutionProvider',
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
    feature_buffer_name = ort_session.get_inputs()[2].name
    feature_buffer_mask_name = ort_session.get_inputs()[3].name

    # image = np.random.rand(1, 3, 256, 256).astype(np.float32)
    # routed_map = np.random.rand(1, 3, 128, 128).astype(np.float32)
    # feature_buffer = np.random.rand(1, 4, 512).astype(np.float32)
    # feature_buffer_mask = np.zeros((1, 4), dtype=np.bool)
    
    # for i in range(100):
    #     start_time = time.time()
    #     ort_inputs = {input_name: image, route_name: routed_map, feature_buffer_name: feature_buffer, feature_buffer_mask_name: feature_buffer_mask}
    #     pred_traj, current_feature = ort_session.run(None, ort_inputs)
    #     pos_x = pred_traj[0][:, 0]
    #     pos_y = pred_traj[0][:, 1]
    #     robot_yaw = pred_traj[0][:, 2]
    #     robot_vel = pred_traj[0][:, 3]

    #     # for i in range(pos_x.shape[0]):
    #     #     x_i = pos_x[i]
    #     #     y_i = pos_y[i]
    #     #     yaw_i = robot_yaw[i]
    #     #     vel_i = robot_vel[i]
    #     #     state = State(x=round(x_i,5),y=round(y_i,5),yaw=round(yaw_i,5),linear_x_vel=round(vel_i,5))
    #     #     states_msg.states.append(state)
            
    #     print(f'FPS: {states_msg}')
    
    rate = rospy.Rate(AI_RATE) # change 30 to 60
    # # Create a mesh object
    count = 0
    crr_time = -1
    count_dupplicate = 0
    start_idx = True

    old_time = -1
    start_dupplicate = time.time()
    stuck_threshold = 0.15

    queue = deque(maxlen=4)

    feature_buffer = np.zeros((1, 4, 512)).astype(np.float32)
    feature_buffer_mask = np.zeros((1, 4), dtype=np.bool)

    feature_count = 0
    
    last_av = 0
    last_lv = 0

    while not rospy.is_shutdown():
        # For each new grab, mesh data is updated        

        # get image from sub
        image = steering_predictor.image
        routed_map = steering_predictor.routed_map

        # image = np.random.randint(255, size=(RESIZE_DIM[0], RESIZE_DIM[1], 3),dtype=np.uint8)
        # routed_map = np.random.randint(255, size=(160, 160, 3),dtype=np.uint8)

        # if image is None or routed_map is None:
        if not steering_predictor.image_flag:
            # print(f"Waiting for image here!")
            time.sleep(0.01)    
            continue

        if not steering_predictor.map_flag:
            if (time.time() - steering_predictor.map_time) > 0.5:
                print(f"------------- Waiting for routed map here!")
                time.sleep(1)    
                continue

        lane_follow_msg.stuck_camera = 0
        
        start_time = time.time()

        H, W, _ = image.shape
        image = cv2.resize(image, (256,256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        image = normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0).astype(np.float32)

        routed_map = routed_map[80-64:80+64, 80-64:80+64]
        routed_map = cv2.cvtColor(routed_map, cv2.COLOR_BGR2RGB)

        # convert new map to old map color
        routed_map = convert_map_color(routed_map)

        routed_map = routed_map / 255.0
        routed_map = np.transpose(routed_map, [2,0,1]).astype(np.float32)
        routed_map = np.expand_dims(routed_map, axis=0).astype(np.float32)

        ort_inputs = {input_name: image, route_name: routed_map, feature_buffer_name: feature_buffer, feature_buffer_mask_name: feature_buffer_mask}
        pred_traj, current_feature = ort_session.run(None, ort_inputs)

        pos_x = pred_traj[0][:, 0]
        pos_y = pred_traj[0][:, 1]
        robot_yaw = pred_traj[0][:, 2]
        robot_vel = pred_traj[0][:, 3]

        states_msg = StateArray()
        path_msg = Path()
        path_msg.header.frame_id = "base_link"

        for i in range(pos_x.shape[0]):
            x_i = pos_x[i]
            y_i = pos_y[i]
            yaw_i = robot_yaw[i]
            vel_i = robot_vel[i]
            state = State(x=round(x_i,5),y=round(y_i,5),yaw=round(yaw_i,5),linear_x_vel=round(vel_i,5))
            states_msg.states.append(state)
            # view in Rviz
            pose = PoseStamped()
            pose.pose.position.x = pos_x[i]
            pose.pose.position.y = pos_y[i]
            qx,qy,qz,qw = get_quaternion_from_euler(0,0,robot_yaw[i])
            pose.pose.orientation.x = qx 
            pose.pose.orientation.y = qy 
            pose.pose.orientation.z = qz 
            pose.pose.orientation.w = qw 
            path_msg.poses.append(pose)

        # states_msg.states = [State(x=round(x_i, 5), y=round(y_i, 5), yaw=round(yaw_i, 5), linear_x_vel=round(vel_i, 5)) for x_i, y_i, yaw_i, vel_i in zip(pos_x, pos_y, robot_yaw, robot_vel)]
        
        # publish
        traj_pub.publish(states_msg)
        path_pub.publish(path_msg)

        feature_count += 1
        if feature_count % 5 == 0:
            feature_buffer = np.concatenate((current_feature, feature_buffer[:, :-1]), axis=1)
            feature_buffer_mask = np.concatenate((np.ones((1,1), dtype=np.bool), feature_buffer_mask[:, :-1]), axis=1)

        final_fps = 1 / (time.time() - start_time)

        # print(f'FPS: {final_fps:.2f}')
        
        print(f'AV: {robot_yaw} | LV: {robot_vel} | FPS: {final_fps:.2f}')

        if feature_count == 99:
            feature_count = 0

        steering_predictor.image_flag = False
        steering_predictor.map_flag = False

        rate.sleep()

if __name__ == "__main__":
    main()
