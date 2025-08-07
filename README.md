# Depth Estimation from Monocular Fisheye Camera (Intel T265)

## Introduction

This repository provides a ROS 2 package for monocular depth estimation and 3D point cloud reconstruction using a single fisheye lens of the Intel RealSense T265 camera. The system uses a transformer-based deep learning model trained on indoor environments and adapted to equirectangular projections to estimate depth maps from wide-FOV fisheye images. The estimated depth is used to generate a dense, colored point cloud in spherical coordinates.

## DepthEstimator Class Overview

### Main `__init__` Variables

* `self.model`: Transformer-based model loaded from a PyTorch `.pt` file
* `self.device`: Target device for inference (GPU/CPU)
* `self.cam_params`: Dictionary with T265 calibration intrinsics and distortion parameters
* `self.erp_size`: Resolution of the equirectangular projection (ERP)
* `self.dummy_depth`: Synthetic depth used for ERP remapping
* `self.latitude_point_cloud`, `self.longitude_point_cloud`: Coordinates used to project ERP pixels to 3D

### Model Initialization

```python
self.model, self.device = self.depth_model_initialization(MODEL_PATH)
```

The model is based on a transformer architecture, specifically adapted to process equirectangular inputs. It was trained using indoor datasets like ScanNet++ and optimized for environments with depth ranges up to approximately 10 meters.

### ERP Projection Method

```python
erp_result = self.erp_projection_transformation(image)
```

* Projects fisheye RGB image into an ERP (equirectangular) view
* Applies dummy depth to guide spatial remapping
* Outputs preprocessed RGB image, attention mask, lat/long tensors for the model

### Depth Estimation Method

```python
def get_depth_image(self, depth_estimation):
```

* Normalizes raw depth values to \[0, 255]
* Applies a colormap (`COLORMAP_INFERNO`) to enhance visualization
* Returns a color-encoded depth image (OpenCV format)

### Point Cloud Reconstruction

```python
def reconstruct_pcd_erp(self, depth, mask):
```

* Converts ERP depth map into a 3D point cloud using latitude/longitude grids
* Applies an optional mask to remove invalid areas
* Returns a (H, W, 3) matrix of XYZ points

### Estimation Pipeline

```python
def estimation_pipeline(self, image):
```

* Full inference process:

  1. Projects input image to ERP
  2. Preprocesses and normalizes image
  3. Performs inference with transformer model
  4. Returns depth map, attention mask, and ERP image tensor

## Depth Node

The `DepthEstimatorNode` is the main ROS 2 node. It performs the following steps:

* Subscribes to `/rs_t265/fisheye_left`
* Calls `DepthEstimator.estimation_pipeline()` on each frame
* Publishes:

  * Color-mapped depth image to `/rs_t265/depth_estimator_node`
  * Colored `PointCloud2` message to `/rs_t265/pointcloud`
* Supports saving `.ply` point cloud file if desired

## Dependencies

```bash
# Create and activate virtual environment
conda create -n dac310 python=3.10
conda activate dac310

# Install Python dependencies
pip install -r requirements.txt
```

## ROS 2 Usage

1. Activate the environment and source ROS setup:

```bash
conda activate dac310
export PYTHONPATH=$PYTHONPATH:/home/rcasal/ros2_ws/src/depth_estimation_dac/scripts
source install/setup.zsh
```

2. Run the required ROS 2 nodes:

```bash
# Run camera node (example)
ros2 run depth_estimation_dac t265_node

# Run depth estimation node
ros2 run depth_estimation_dac depth_node.py
```

3. (Alternative) Use launch file:

```bash
ros2 launch depth_estimation_dac depth_launch.py
```
