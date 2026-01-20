#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageProcessing(Node):

    def __init__(self):
        super().__init__('watershed_node')
        self.sub = self.create_subscription(Image, "/transformed_image", self.image_callback, 10)
        self.pub = self.create_publisher(Image, "/final_image", 10)
        self.bridge = CvBridge()
        self.transformed_image = None

    def image_callback(self, msg):
        self.transformed_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")


def main(args=None):
    rclpy.init(args=args)
    fb = ImageProcessing()

    while rclpy.ok():
        rclpy.spin_once(fb)

        if fb.transformed_image is not None:
            image = fb.transformed_image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), sigmaX=0, sigmaY=0)
            sigma = np.std(gray)
            mean = np.mean(gray)
            lower = int(max(0, (mean - sigma)))
            upper = int(min(255, (mean + sigma)))
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            thresh_inverted = cv2.bitwise_not(thresh)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh_inverted, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 0] = 0
            markers = cv2.watershed(image, markers)
            image[markers == -1] = [0, 255, 0]
            segmented_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(segmented_gray, lower, upper)
            edges = cv2.dilate(edges,kernel,iterations = 1)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 100  
            filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
            contour_image = np.copy(image)
            
            for contour in filtered_contours:
                cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 1)
            cv2.imshow('Blurred image', blurred)
            cv2.imshow('Segmented Image using Watershed', image)
            cv2.imshow('Canny Edges', edges)
            cv2.imshow('Filtered Contours on Image', contour_image)
            cv2.waitKey(1)
        
            if edges is not None:
                contour_img_msg = fb.bridge.cv2_to_imgmsg(edges, "mono8")
                fb.pub.publish(contour_img_msg)

    fb.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()