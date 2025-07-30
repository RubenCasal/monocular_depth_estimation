import torch
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import torchvision.transforms.functional as TF
import time
import onnxruntime as ort
import torch.nn.functional as F
import open3d as o3d
import os

import sys
sys.path.append(os.path.dirname(__file__))

from utils import cam_to_erp_patch_fast
from utils import resize_for_input


class DepthEstimator:
    def __init__(self):

        # Model Initialization
     
        MODEL_PATH = "/home/rcasal/ros2_ws/src/depth_estimation_dac/scripts/dac_model_full.pt"

        self.model, self.device = self.depth_model_initialization(MODEL_PATH)

        # Camera calibration parameters
        self.cam_params =  {
            "dataset": "scannetpp",
            "fl_x": 285.720703125,
            "fl_y": 285.427490234375,
            "fx": 285.720703125,
            "fy": 285.427490234375,
            "cx": 411.068206787109,
            "cy": 394.509887695312,
            "k1": -0.0046993619762361,
            "k2": 0.0400081910192966,
            "k3": -0.037823498249054,
            "k4": 0.00574744818732142,
            "camera_model": "OPENCV_FISHEYE"
        }

        self.cano_sz = (512,512)
        self.crop_w_fov = 180
        self.fwd_sz = (512, 750)
        self.erp_size = (512,750)

        # Image (input) parameters
        self.input_image_shape = (800, 848)
        self.batch_size = 1
        self.lat_range = torch.tensor([[-np.pi / 2, np.pi / 2]], dtype=torch.float32).repeat(self.batch_size, 1).to(self.device)
        self.long_range = torch.tensor([[-np.pi, np.pi]], dtype=torch.float32).repeat(self.batch_size, 1).to(self.device)

        self.latitude_point_cloud = np.linspace(-np.pi / 2, np.pi / 2, self.erp_size[0])
        self.longitude_point_cloud = np.linspace(np.pi, 0, self.erp_size[1])

        self.normalization_stats = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

       
        dummy_depth = np.ones((self.input_image_shape[0],self.input_image_shape[1]), dtype=np.float32)
        x, y = np.meshgrid(np.arange(dummy_depth.shape[1]), np.arange(dummy_depth.shape[0]))
        fx = self.cam_params["fx"]
        cx = self.cam_params["cx"]
        cy = self.cam_params["cy"]
        dummy_depth *= np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + fx ** 2) / fx

        self.dummy_depth = np.expand_dims(dummy_depth, axis=2)
        self.mask_valid_depth = self.dummy_depth > 0.01

        # Compute ERP crop dimensions
        self.crop_width = int(self.cano_sz[0] * self.crop_w_fov / 180)
        self.crop_height = int(self.crop_width * self.fwd_sz[0] / self.fwd_sz[1])

        self.phi = np.array(0).astype(np.float32)
        self.roll = np.array(0).astype(np.float32)
        self.theta = 0

        self.intrinsic_dummy = np.eye(3, dtype=np.float32)



    def depth_model_initialization(self, model_path):

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = torch.load(model_path, map_location=device)

        print(f"The model has been initialized with device: {device}")

        return model, device
    

    def erp_projection_transformation(self, image):
        """
        Applies ERP projection and preprocessing using dummy depth for inference.

        Args:
            image (np.array): RGB image HxWx3 (uint8 or float32).
            sample_json (dict): Contains cam_params and possibly fisheye grid.
            cano_sz (tuple): Canonical size (default: 512x512).
            fwd_sz (tuple): Model forward input size (default for Swin-L: 500x750).

        Returns:
            dict: preprocessed image, lat/long range, attention mask, scale factor, padding.
        """

        # Ensure image is float32 in range [0,1]
        image = image.astype(np.float32) / 255.0

        # ERP projection
        image_erp, depth_erp, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image, self.dummy_depth, (self.mask_valid_depth * 1.0).astype(np.float32), self.theta, self.phi,
            self.crop_height, self.crop_width, self.cano_sz[0], self.cano_sz[0]*2,
            self.cam_params, self.roll, scale_fac=None
        )

        image_resized, depth_resized, pad, scale_factor, attn_mask = resize_for_input(
            (image_erp * 255.).astype(np.uint8),
            depth_erp,
            self.fwd_sz,
            self.intrinsic_dummy,
            canonical_shape=[image_erp.shape[0], image_erp.shape[1]],
            to_canonical_ratio=1.0,
            padding_rgb=[0, 0, 0],
            mask=erp_mask
        )

        lat_range = torch.tensor([float(np.min(latitude)), float(np.max(latitude))])
        long_range = torch.tensor([float(np.min(longitude)), float(np.max(longitude))])

      
        return {
            "image": image_resized,
            "mask": attn_mask,
            "pad": pad,
            "scale_factor": scale_factor,
            "lat_range": lat_range,
            "long_range": long_range
        }
    
    def pass_to_tensor(self,image):
        image = TF.normalize(TF.to_tensor(image), **self.normalization_stats)
        return image.unsqueeze(0)
    
    def model_inference(self, image, model, lat_range, long_range, pred_scale_factor, device):
        lat_range = lat_range.unsqueeze(0).to(device)  
        long_range = long_range.unsqueeze(0).to(device)
        with torch.no_grad():
            preds, _, _ = model(image.to(device), lat_range, long_range)

        preds *= pred_scale_factor
        return preds
    

    def display_depth(self, depth_estimation):
        depth_vis = (depth_estimation - depth_estimation.min()) / (depth_estimation.max() - depth_estimation.min())

        # Display using matplotlib
        plt.imshow(depth_vis, cmap='inferno')  # or 'inferno', 'magma', 'viridis'
        plt.colorbar(label='Depth')
        plt.title("Predicted Depth Map")
        plt.axis('off')
        plt.show()
    

    def get_depth_image(self, depth_estimation):
 
        depth_vis = (depth_estimation - depth_estimation.min()) / (depth_estimation.max() - depth_estimation.min())
        depth_vis = (depth_vis * 255).astype(np.uint8)

        print(depth_estimation.shape)

        # Aplica colormap (opcional, pero da buena visualización)
        depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        return depth_vis_color 

        

    def reconstruct_pcd_erp(self, depth, mask):
        "Assume depth in euclid distance"

        if type(depth) == torch.__name__:
            depth = depth.cpu().numpy().squeeze()
        depth = cv2.medianBlur(depth, 5)

        # Assume to use hemishperes of 360 degree camera if no range is given
        
        longitude, latitude = np.meshgrid(self.longitude_point_cloud, self.latitude_point_cloud)
        longitude = longitude.astype(np.float32)
        latitude = latitude.astype(np.float32)

        x = -np.cos(latitude) * np.cos(longitude)
        z = np.cos(latitude) * np.sin(longitude)
        y = np.sin(latitude)
        
        
        pcd_base = np.concatenate([x[:, :, None], y[:, :, None], z[:, :, None]], axis=2)
        pcd = depth[:, :, None] * pcd_base
        # if mask is not None:
        #     pcd[mask] = 0
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy().squeeze().astype(bool)
            mask = mask.astype(bool)
            pcd[~mask] = 0
        return pcd
    
    def save_xyz_ply(self,points_xyz, filename="pointcloud_xyz.ply"):
        """
        Save a 3D point cloud (XYZ only) as a .ply file (ASCII format).
        
        Args:
            points_xyz: (N, 3) ndarray of float32 (x, y, z)
            filename: output .ply path
        """
        # Filter invalid (0,0,0) points
        valid_mask = ~np.all(points_xyz == 0, axis=1)
        points_xyz = points_xyz[valid_mask]

        with open(filename, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_xyz)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for pt in points_xyz:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

    def save_colored_ply(self, points, colors, filename="colored_pointcloud.ply"):
        """
        points: (N, 3) array of XYZ
        colors: (N, 3) array of RGB in [0, 255]
        """
        assert points.shape == colors.shape

        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


    def save_file_ply(self, xyz, rgb, pc_file):
        if rgb.max() < 1.001:
            rgb = rgb * 255.0
        rgb = rgb.astype(np.uint8)
        with open(pc_file, "w") as f:
            # headers
            f.writelines(
                [
                    "ply\n" "format ascii 1.0\n",
                    "element vertex {}\n".format(xyz.shape[0]),
                    "property float x\n",
                    "property float y\n",
                    "property float z\n",
                    "property uchar red\n",
                    "property uchar green\n",
                    "property uchar blue\n",
                    "end_header\n",
                ]
            )

            for i in range(xyz.shape[0]):
                str_v = "{:10.6f} {:10.6f} {:10.6f} {:d} {:d} {:d}\n".format(
                    xyz[i, 0], xyz[i, 1], xyz[i, 2], rgb[i, 0], rgb[i, 1], rgb[i, 2]
                )
                f.write(str_v)

    def process_rgb_point_cloud(self,
    rgb 

    ):
    
        mean = np.array([123.675, 116.28, 103.53])[:, np.newaxis, np.newaxis]
        std= np.array([58.395, 57.12, 57.375])[:, np.newaxis, np.newaxis]

    
        rgb = ((rgb * std) + mean).astype(np.uint8)
        rgb = rgb.transpose((1, 2, 0))
        
        return rgb



    def create_pointcloud2_msg(points, colors, frame_id="camera_link"):
        # points: (N,3) float32
        # colors: (N,3) uint8

        # Convert RGB to packed float32
        rgb_packed = (
            (colors[:, 0].astype(np.uint32) << 16)
            | (colors[:, 1].astype(np.uint32) << 8)
            | colors[:, 2].astype(np.uint32)
        )
        rgb_packed = rgb_packed.view(np.float32)

        # Combine XYZ + RGB
        cloud_data = np.column_stack((points, rgb_packed)).astype(np.float32)

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]

        header = std_msgs.msg.Header()
        header.stamp = rclpy.time.Time().to_msg()
        header.frame_id = frame_id

        return pc2.create_cloud(header, fields, cloud_data)
    
    def estimation_pipeline(self, image):
        erp_result = self.erp_projection_transformation(image)

        image = erp_result["image"]
          
        lat_range = erp_result["lat_range"]
        long_range = erp_result["lat_range"]
        pred_scale_factor = erp_result["scale_factor"]

       
        image_tf = self.pass_to_tensor(image)
       
        if image_tf.dim() == 3:
            image_tf = image_tf.unsqueeze(0)  # (1, 3, H, W)

        print(f"Dimension del tensor {image_tf.shape}")
        image_tf = F.interpolate(image_tf, size=self.erp_size, mode='bilinear', align_corners=False)


        preds = self.model_inference(image_tf, self.model, lat_range, long_range, pred_scale_factor, self.device)

        depth_np = preds.squeeze().detach().cpu().numpy()

        return depth_np, erp_result["mask"], image_tf



