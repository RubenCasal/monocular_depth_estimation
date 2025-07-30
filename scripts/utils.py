import torch
import numpy as np
import math
import torch.nn.functional as F
import cv2

def cam_to_erp_patch_fast(img, depth, mask_valid_depth, theta, phi, patch_h, patch_w, erp_h, erp_w, cam_params, roll=None, scale_fac=None, padding_rgb=[123.675, 116.28, 103.53]):
    """
        This is an efficient implementation in two folds:
            - Only consider coordinates within target ERP patch
            - Implement using explicit Gnomonic Projection -- https://mathworld.wolfram.com/GnomonicProjection.html
            
        
        Args:
            img: the source perspective image [img_h, img_w, 3]
            depth: the corresponding depth map [img_h, img_w, 1]
            mask_valid_depth: the valid depth mask [img_h, img_w, 1]
            theta: the longitude of the target patch center
            phi: the latitude of the target patch center
            patch_h: the height of the target patch
            patch_w: the width of the target patch
            erp_h: the height of the whole equirectangular projection
            erp_w: the width of the whole equirectangular projection
            cam_params: the camera parameters, check the usage in code for details
            roll: the camera roll angle in radians
            scale_aug: the scale augmentation factor, 0 means no augmentation
        output:
            erp_img: the target patch in equirectangular projection [3, patch_h, patch_w]
            erp_depth: the corresponding depth in equirectangular projection [patch_h, patch_w]
            erp_mask_valid_depth: the valid depth mask in equirectangular projection [patch_h, patch_w]
            mask_active: the mask indicating the valid area in the target patch [patch_h, patch_w]
            lat_grid: the latitude grid in the target patch [patch_h, patch_w]
            lon_grid: the longitude grid in the target patch [patch_h, patch_w]
    """
    [img_h, img_w, _] = img.shape

    img_new = np.transpose(img, [2, 0, 1])
    img_new = torch.from_numpy(img_new).unsqueeze(0)
    depth_new = np.transpose(depth, [2, 0, 1])
    depth_new = torch.from_numpy(depth_new).unsqueeze(0)
    mask_valid_depth = np.transpose(mask_valid_depth, [2, 0, 1])
    mask_valid_depth = torch.from_numpy(mask_valid_depth).unsqueeze(0)

    PI = math.pi
    PI_2 = math.pi * 0.5
    PI2 = math.pi * 2
    # compute the target FOV based on target patch size and whole erp size
    wFOV_tgt = patch_w / erp_w * PI2
    hFOV_tgt = patch_h / erp_h * PI

    # only target patch erp coordinates
    cp = torch.tensor([theta, phi]).view(1, 1, -1)
    lat_grid, lon_grid = torch.meshgrid(
        torch.linspace(phi - hFOV_tgt/2, phi + hFOV_tgt/2, patch_h),
        torch.linspace(theta - wFOV_tgt/2, theta + wFOV_tgt/2, patch_w))
    lon_grid = lon_grid.float().reshape(1, -1)  # .repeat(num_rows*num_cols, 1)
    lat_grid = lat_grid.float().reshape(1, -1)  # .repeat(num_rows*num_cols, 1)
        
    # TODO: lat_grid may need cap to -pi/2 and pi/2 if crop size is large and pitch angle is large
    # compute corresponding perp image coordinates via Gnomonic Project explicitly (x, y given sphere radius 1 or f=1 perspective image)
    cos_c = torch.sin(cp[..., 1]) * torch.sin(lat_grid) + torch.cos(cp[..., 1]) * torch.cos(lat_grid) * torch.cos(
        lon_grid - cp[..., 0])
    x_num = (torch.cos(lat_grid) * torch.sin(lon_grid - cp[..., 0]))
    y_num = (torch.cos(cp[..., 1]) * torch.sin(lat_grid) - torch.sin(cp[..., 1]) * torch.cos(lat_grid) * torch.cos(
        lon_grid - cp[..., 0]))
    new_x = x_num / cos_c
    new_y = y_num / cos_c

    # OPTIONAL: apply camera roll correction
    if roll is not None:
        roll = torch.tensor(roll, dtype=torch.float32)
        new_x_tmp = new_x * torch.cos(roll) - new_y * torch.sin(roll)
        new_y_tmp = new_x * torch.sin(roll) + new_y * torch.cos(roll)
        new_x = new_x_tmp
        new_y = new_y_tmp

    # Scale augmentation can just modify new_x and new_y directly, with depth factor
    if scale_fac is not None:
        new_x *= scale_fac
        new_y *= scale_fac
        depth_new *= scale_fac # depth value is adjusted for the scale augmentation
        # print(scale_fac)

    # Important: this normalization needs to use the source perspective image FOV, for the grid sample function range [-1, 1]
    # Gnomonic Projection for OPENCV_FISHEYE model, only works for FOV < 180 degree
    if 'camera_model' in cam_params.keys() and cam_params['camera_model'] == 'OPENCV_FISHEYE':    
        """
            Apply opencv distortion (Refer to: OpenCV)
        """
        k1 = cam_params['k1']
        k2 = cam_params['k2']
        k3 = cam_params['k3']
        k4 = cam_params['k4']
        fx = cam_params['fl_x']
        fy = cam_params['fl_y']
        cx = cam_params['cx']
        cy = cam_params['cy']

        # Option 1: original opencv fisheye distortion can not handle FOV >=180 degree or cos_c <= 0
        # r = np.sqrt(new_x*new_x + new_y*new_y)
        # theta = np.arctan(r)
        # theta_d = theta * (1 + k1*theta*theta + k2*theta**4 + k3*theta**6 + k4*theta**8)        
        # x_d = theta_d * new_x / (r+1e-9)
        # y_d = theta_d * new_y / (r+1e-9)

        # Option 2: A more numerically stable version able to handle FOV >=180 degree, adapted for Gnomonic Projection
        r = np.sqrt(x_num*x_num + y_num*y_num)
        theta = np.arccos(cos_c)
        theta_d = theta * (1 + k1*theta*theta + k2*theta**4 + k3*theta**6 + k4*theta**8)
        x_d = theta_d * x_num / (r)
        y_d = theta_d * y_num / (r)
        
        # project to image coordinates
        new_x = fx * x_d + cx
        new_y = fy * y_d + cy
        
        """
            Projection to image coordinates using intrinsic parameters
        """
        new_x -= img_w/2
        new_x /= (img_w/2)
        new_y -= img_h/2
        new_y /= (img_h/2)
        
    # Gnomonic Projection for MEI model, but only works for FOV < 180 degree (kitti360 is slightly beyond 180)
    elif 'camera_model' in cam_params.keys() and cam_params['camera_model'] == 'MEI':
        xi = cam_params['xi']
        k1 = cam_params['k1']
        k2 = cam_params['k2']
        p1 = cam_params['p1']
        p2 = cam_params['p2']
        fx = cam_params['fx']
        fy = cam_params['fy']
        cx = cam_params['cx']
        cy = cam_params['cy']
        
        # Adpated for Gnomonic Projection
        p_u = x_num / (cos_c + xi)
        p_v = y_num / (cos_c + xi)

        # apply distortion
        ro2 = p_u*p_u + p_v*p_v

        p_u *= 1 + k1*ro2 + k2*ro2*ro2
        p_v *= 1 + k1*ro2 + k2*ro2*ro2

        p_u += 2*p1*p_u*p_v + p2*(ro2 + 2*p_u*p_u)
        p_v += p1*(ro2 + 2*p_v*p_v) + 2*p2*p_u*p_v

        # apply projection
        new_x = fx*p_u + cx 
        new_y = fy*p_v + cy
        
        """
            Projection to image coordinates using intrinsic parameters
        """
        new_x -= img_w/2
        new_x /= (img_w/2)
        new_y -= img_h/2
        new_y /= (img_h/2)
        
    # elif cam_params['dataset'] == 'nyu': # uncomment if you want to consider nyu as fisheye, very slight distortion
    #     kc = [cam_params['k1'], cam_params['k2'], cam_params['p1'], cam_params['p2'], cam_params['k3']]
    #     fx = cam_params['fx']
    #     fy = cam_params['fy']
    #     cx = cam_params['cx']
    #     cy = cam_params['cy']
    #
    #     """
    #         Apply distortion (Refer to: NYUv2 Toolbox and http://www.vision.caltech.edu/bouguetj/calib_doc/)
    #     """
    #     r_sq = new_x**2 + new_y**2
    #     dx = 2*kc[2]*new_x*new_y + kc[3]*(r_sq + 2*new_x**2)
    #     dy = kc[2]*(r_sq + 2*new_y**2) + 2*kc[3]*new_x*new_y
    #     new_x = (1 + kc[0]*r_sq + kc[1]*r_sq**2 + kc[4]*r_sq**3)*new_x + dx
    #     new_y = (1 + kc[0]*r_sq + kc[1]*r_sq**2 + kc[4]*r_sq**3)*new_y + dy

    #     """
    #         Projection to image coordinates using intrinsic parameters
    #     """
    #     new_x = fx * new_x + cx
    #     new_y = fy * new_y + cy

    #     # convert to grid_sample range [-1, 1] scope (could extend due to larger ERP range or shifted principle center)
    #     new_x -= img_w/2
    #     new_x /= (img_w/2)
    #     new_y -= img_h/2
    #     new_y /= (img_h/2)
    else:
        # If necessuary, handle principal point shift in perspective data (e.g., KITTI, DDAD, LYFT)
        if 'cx' in cam_params.keys():
            new_x = cam_params['fx'] * new_x + cam_params['cx']
            new_y = cam_params['fy'] * new_y + cam_params['cy']
            # convert to grid_sample range [-1, 1] scope (could extend due to larger ERP range or shifted principle center)
            new_x -= img_w/2
            new_x /= (img_w/2)
            new_y -= img_h/2
            new_y /= (img_h/2)
        else:    
            # assume FOV in radians
            new_x = new_x / np.tan(cam_params['wFOV'] / 2)
            new_y = new_y / np.tan(cam_params['hFOV'] / 2)

    new_x = new_x.reshape(1, patch_h, patch_w)
    new_y = new_y.reshape(1, patch_h, patch_w)
    new_grid = torch.stack([new_x, new_y], -1)

    # those value within -1, 1 corresponding to content area
    mask_active = torch.logical_and(
        torch.logical_and(new_x > -1, new_x < 1),
        torch.logical_and(new_y > -1, new_y < 1),
    )*1.0

    # inverse mapping through grid_sample function in pytorch. Alternative is cv2.remap
    erp_img = F.grid_sample(img_new, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
    erp_img *= mask_active

    # compute depth in erp
    erp_depth = F.grid_sample(depth_new, new_grid, mode='nearest', padding_mode='border', align_corners=True)
    erp_depth *= mask_active

    # compute the valid depth mask in erp
    erp_mask_valid_depth = F.grid_sample(mask_valid_depth, new_grid, mode='nearest', padding_mode='border', align_corners=True)
    erp_mask_valid_depth *= mask_active

    # output
    erp_img = erp_img[0].permute(1, 2, 0).numpy()
    erp_depth = erp_depth[0, 0].numpy().astype(np.float32)
    erp_mask_valid_depth = erp_mask_valid_depth[0, 0].numpy().astype(np.float32)
    mask_active = mask_active[0].numpy().astype(np.float32)
    lat_grid = lat_grid.reshape(patch_h, patch_w).numpy().astype(np.float32)
    lon_grid = lon_grid.reshape(patch_h, patch_w).numpy().astype(np.float32)
    
    # # apply invalid region to padding_rgb
    # erp_img[erp_mask_valid_depth == 0] = np.array(padding_rgb)/255
    # erp_depth[mask_active==0] = np.array(padding_rgb[0])/255
    
    return erp_img, erp_depth, erp_mask_valid_depth, mask_active, lat_grid, lon_grid

def resize_for_input(image, depth, output_shape, intrinsic, canonical_shape, to_canonical_ratio, padding_rgb=[123.675, 116.28, 103.53], mask=None):
    """
    Resize the input.
    Resizing consists of two processed, i.e. 1) to the canonical space (adjust the camera model); 2) resize the image while the camera model holds. Thus the
    label will be scaled with the resize factor.
    
    If the image is the original image, just set to_canonical_ratio=1, canonical_shape as the original image shape.
    """

    h, w, _ = image.shape
    resize_ratio_h = output_shape[0] / canonical_shape[0]
    resize_ratio_w = output_shape[1] / canonical_shape[1]
    to_scale_ratio = min(resize_ratio_h, resize_ratio_w)

    resize_ratio = to_canonical_ratio * to_scale_ratio

    reshape_h = int(resize_ratio * h)
    reshape_w = int(resize_ratio * w)

    pad_h = max(output_shape[0] - reshape_h, 0)
    pad_w = max(output_shape[1] - reshape_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    # resize
    image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
    # padding
    image = cv2.copyMakeBorder(
        image, 
        pad_h_half, 
        pad_h - pad_h_half, 
        pad_w_half, 
        pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, 
        value=padding_rgb)
    depth = cv2.copyMakeBorder(
        depth, 
        pad_h_half, 
        pad_h - pad_h_half, 
        pad_w_half, 
        pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, 
        value=0)
    
    # This might be wrong, the longer side may not lead to center position by multiplying min(resize_ratio_h, resize_ratio_w). The padding should be included
    # intrinsic[0, 2] = intrinsic[0, 2] * to_scale_ratio
    # intrinsic[1, 2] = intrinsic[1, 2] * to_scale_ratio

    pad=[pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    # label_scale_factor=1/to_scale_ratio
    
    if mask is not None:
        mask = cv2.resize(mask, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.copyMakeBorder(
            mask, 
            pad_h_half, 
            pad_h - pad_h_half, 
            pad_w_half, 
            pad_w - pad_w_half, 
            cv2.BORDER_CONSTANT, 
            value=0)
        return image, depth, pad, to_scale_ratio, mask
    
    return image, depth, pad, to_scale_ratio