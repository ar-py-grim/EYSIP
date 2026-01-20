import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import signal
import sys

class LidarComparisonNode(Node):

    def __init__(self):
        super().__init__('lidar_comparison_node')

        self.subscription_scan = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile_sensor_data)

        self.subscription_vscan = self.create_subscription(
            LaserScan,
            'vscan',
            self.vscan_callback,
            qos_profile_sensor_data)

        self.scan_data = None
        self.vscan_data = None
        self.real_lidar_data = []
        self.virtual_lidar_data = []

    def scan_callback(self, msg):
        self.scan_data = (np.array(msg.ranges), msg.angle_min, msg.angle_increment)
        self.real_lidar_data.append(self.scan_data[0])
        self.compare_lidar_data()
        self.plot_scans()

    def vscan_callback(self, msg):
        self.vscan_data = (np.array(msg.ranges), msg.angle_min, msg.angle_increment)
        self.virtual_lidar_data.append(self.vscan_data[0])
        self.compare_lidar_data()
        self.plot_scans()

    def compare_lidar_data(self):
        if self.scan_data is not None and self.vscan_data is not None:
            real_data = self.scan_data[0]
            virtual_data = self.vscan_data[0]
            if real_data.shape == virtual_data.shape:
                rmse = np.sqrt(mean_squared_error(real_data, virtual_data))
                mae = mean_absolute_error(real_data, virtual_data)
                self.get_logger().info(f'Accuracy (RMSE): {rmse}')
                self.get_logger().info(f'Accuracy (MAE): {mae}')

                if len(self.real_lidar_data) > 1 and len(self.virtual_lidar_data) > 1:
                    real_stdev = np.std(self.real_lidar_data, axis=0)
                    virtual_stdev = np.std(self.virtual_lidar_data, axis=0)
                    self.get_logger().info(f'Precision (2D LiDAR): {np.mean(real_stdev)}')
                    self.get_logger().info(f'Precision (Virtual LiDAR): {np.mean(virtual_stdev)}')

    def plot_scans(self):
        if self.scan_data is not None and self.vscan_data is not None:
            scan_ranges, scan_angle_min, scan_angle_increment = self.scan_data
            vscan_ranges, vscan_angle_min, vscan_angle_increment = self.vscan_data

            scan_angles = scan_angle_min + np.arange(len(scan_ranges)) * scan_angle_increment
            vscan_angles = vscan_angle_min + np.arange(len(vscan_ranges)) * vscan_angle_increment

            scan_x = scan_ranges * np.cos(scan_angles)
            scan_y = scan_ranges * np.sin(scan_angles)

            vscan_x = vscan_ranges * np.cos(vscan_angles)
            vscan_y = vscan_ranges * np.sin(vscan_angles)

            plt.figure()
            plt.scatter(scan_x, scan_y, label='scan', s=1)
            plt.scatter(vscan_x, vscan_y, label='vscan', s=1, alpha=0.6)
            plt.title('Lidar Scan Visualization')
            plt.xlabel('X (meters)')
            plt.ylabel('Y (meters)')
            plt.legend()
            plt.axis('equal')
            plt.show()

            plt.close('all')
            rclpy.shutdown()

def signal_handler(sig, frame):
    plt.close('all')
    rclpy.shutdown()
    sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    lidar_comparison_node = LidarComparisonNode()
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        rclpy.spin(lidar_comparison_node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')
        lidar_comparison_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
