#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math
from geometry_msgs.msg import Pose2D
from utility import *

class FeedBack(Node):

    def __init__(self):
        super().__init__('feedback_node')
        
        # Subscribe to the topic /camera/image_raw to receive image data.
        self.sub = self.create_subscription(Image, "/camera1/image_raw", self.image_callback, 10)
        
        # Create a publisher for the detected ArUco marker's pose.
        self.pub1 = self.create_publisher(Pose2D, "/pen2_pose", 1)

        # Create a publisher for the transformed image
        self.pub2 = self.create_publisher(Image, "/transformed_image", 10)

        # Create a CvBridge instance to convert between ROS image messages and OpenCV images.
        self.bridge = CvBridge()
        
        # Create a Pose2D message to store the detected ArUco marker's pose.
        self.msg_1 = Pose2D()

        self.msg = [self.msg_1]
        self.topleft = None
        self.topright = None
        self.bottomleft = None
        self.bottomright = None
        self.caliberated_image = None
        self.arena_corners = []
        bot_ids = [1]
        self.bot_paths = [[] for _ in range(len(bot_ids))]
        self.distorted_image = None
        self.transformed_image = None
        self.corners = []
        self.ids = []
        self.previous_tL = None
        self.previous_tR = None
        self.previous_bR = None
        self.previous_bL = None
        
        
    def image_callback(self, msg):
        # Convert the ROS image message to an OpenCV image.
        self.distorted_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")    


def main(args=None):
    rclpy.init(args=args)
    fb = FeedBack()
    rclpy.spin_once(fb)
    
    while rclpy.ok():
        bot_ids = [2]
        fb.caliberated_image = camera_caliberation(fb.distorted_image)
        fb.corners, fb.ids = aruco_detect_edge(fb.caliberated_image)
        fb.topleft, fb.topright, fb.bottomleft, fb.bottomright = get_corners(fb.corners, fb.ids)

        if fb.topleft is not None and fb.topright is not None and fb.bottomleft is not None and fb.bottomright is not None:
            fb.arena_corners = np.float32([fb.topleft, fb.topright, fb.bottomleft, fb.bottomright])
            fb.previous_bL = fb.bottomleft
            fb.previous_bR = fb.bottomright
            fb.previous_tL = fb.topleft
            fb.previous_tR = fb.topright

        if fb.bottomleft is None:
            fb.bottomleft = fb.previous_bL  
                                     
        if fb.topright is None:
            fb.topright = fb.previous_tR

        if fb.bottomright is None:
            fb.bottomright = fb.previous_bR

        if fb.topleft is None:
            fb.topleft = fb.previous_tL

        if fb.topleft is not None and fb.topright is not None and fb.bottomleft is not None and fb.bottomright is not None:
            fb.transformed_image = perspective_transform(fb.caliberated_image, fb.arena_corners)
   
        # Check if marker ID 1 is detected.
        if fb.transformed_image is not None:
            corners, ids = aruco_detect_bot(thresholding(fb.transformed_image))
            
            for i in range(len(corners)):
                for j in range(len(bot_ids)):
                    if ids[i][0] == bot_ids[j]:
                        arr = corners[i][0]
                        # Calculate the center coordinates and orientation (theta) of the detected marker
                        centre_x_opencv = sum(arr[:, 0]) / 4
                        centre_y_opencv = sum(arr[:, 1]) / 4
                        
                        # Remove the marker from the image using inpainting with an expanded mask
                        mask = np.zeros(fb.transformed_image.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask, [np.int32(arr)], 255)
                        kernel = np.ones((14, 14), np.uint8)  # Increase the kernel size as needed
                        mask = cv2.dilate(mask, kernel, iterations=5)
                        fb.transformed_image = cv2.inpaint(fb.transformed_image, mask, 
                                                           inpaintRadius= 5,flags= cv2.INPAINT_NS)

                        # Calculate the coordinates in the Cartesian plane
                        centre_x = centre_x_opencv
                        centre_y = 500 - centre_y_opencv

                        # Calculate the orientation angle (theta) of the marker.
                        right_centre_x = (float((arr[1, 0] + arr[2, 0]) / 2))
                        right_centre_y = 500 - float((arr[1, 1] + arr[2, 1]) / 2)            
                        theta = math.atan2((centre_y - right_centre_y), centre_x - right_centre_x)
                        
                        theta += math.pi
                        fb.msg[j].x = centre_x
                        fb.msg[j].y = centre_y
                        fb.msg[j].theta = theta
                        fb.pub1.publish(fb.msg[0])
                                    
                        cv2.imshow('Calibrated Frame', fb.caliberated_image)
                        cv2.imshow('Distorted Frame', fb.distorted_image)
                        cv2.imshow('Aruco removed Frame', fb.transformed_image)
                        cv2.waitKey(1)
            
            fb.transformed_image = cv2.cvtColor(fb.transformed_image, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE (adaptive histogram equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            fb.transformed_image = clahe.apply(fb.transformed_image)

            # Convert the equalized grayscale image back to BGR format
            fb.transformed_image = cv2.cvtColor(fb.transformed_image, cv2.COLOR_GRAY2BGR)
            # Apply white borders to the image
            border_thickness = 10  # Define the thickness of the border
            fb.transformed_image = cv2.copyMakeBorder(fb.transformed_image, border_thickness,
                                                       border_thickness,border_thickness,
                                                      border_thickness, cv2.BORDER_CONSTANT,
                                                      value=[255, 255, 255])
            
            cv2.imshow('CLAHE equalized tformed with borders', fb.transformed_image)
            cv2.waitKey(1)
            transformed_image_msg = fb.bridge.cv2_to_imgmsg(fb.transformed_image, "bgr8")
            fb.pub2.publish(transformed_image_msg)

        rclpy.spin_once(fb)
        
    fb.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()