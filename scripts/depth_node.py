#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from rclpy.qos import qos_profile_sensor_data
import std_msgs.msg
import sensor_msgs_py.point_cloud2 as pc2
import sys

sys.path.append(os.path.dirname(__file__))
from depth_estimator import DepthEstimator


class DepthEstimatorNode(Node):
    def __init__(self):
        super().__init__("depth_node")
        self.bridge = CvBridge()

        # Estimador de profundidad
        self.depth_estimator = DepthEstimator()

        # Subscripción a la imagen de cámara
        self.subscription = self.create_subscription(
            Image,
            "/rs_t265/fisheye_left",
            self.image_callback,
            qos_profile_sensor_data,
        )

        # Publicadores
        self.image_publisher = self.create_publisher(Image, "/rs_t265/depth_estimator_node", 1)
        self.pcd_publisher = self.create_publisher(PointCloud2, "/rs_t265/pointcloud", 1)

        self.get_logger().info("✅ Depth node initialized")

    def image_callback(self, msg):
        try:
            # ROS Image → OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Inferencia
            depth, mask, erp_image = self.depth_estimator.estimation_pipeline(cv_image)
            depth_image = self.depth_estimator.get_depth_image(depth)

            # Publicar imagen de profundidad visualizada
            ros_img_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="bgr8")
            ros_img_msg.header.stamp = self.get_clock().now().to_msg()
            ros_img_msg.header.frame_id = "t265_frame"
            self.image_publisher.publish(ros_img_msg)

            # Reconstrucción de nube de puntos
            pcd = self.depth_estimator.reconstruct_pcd_erp(depth, mask)
            xyz = pcd.reshape(-1, 3)

            rgb_img = erp_image.squeeze(0).cpu().numpy()
            rgb = self.depth_estimator.process_rgb_point_cloud(rgb_img)
            rgb = rgb.reshape(-1, 3)

            # Filtrado: eliminar puntos con z <= 0
            valid_mask = xyz[:, 2] > 0
            xyz = xyz[valid_mask]
            rgb = rgb[valid_mask]

            # Header PointCloud2
            header = std_msgs.msg.Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "t265_frame"  # ⚠️ Compatible con RViz2

            # Crear y publicar mensaje PointCloud2
            pc_msg = self.create_pointcloud2_msg(xyz, rgb, header)
            self.pcd_publisher.publish(pc_msg)

            self.get_logger().info("📦 Published depth image and point cloud")

        except Exception as e:
            self.get_logger().error(f"❌ Failed to process image: {e}")

    def create_pointcloud2_msg(self, points, colors, header):
        # RGB uint8 → packed float32
        rgb_packed = (
            (colors[:, 0].astype(np.uint32) << 16)
            | (colors[:, 1].astype(np.uint32) << 8)
            | colors[:, 2].astype(np.uint32)
        ).astype(np.uint32)

        rgb_packed = rgb_packed.view(np.float32)

        cloud_data = np.column_stack((points, rgb_packed)).astype(np.float32)

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        return pc2.create_cloud(header, fields, cloud_data)


def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
