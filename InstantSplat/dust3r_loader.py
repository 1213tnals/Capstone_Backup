import os 
import torch
from utils.graphics_utils import focal2fov 


def prepare_dust3r_extrinsics(path): 
    camera_extrinsics = [] 
    
    poses = torch.load(os.path.join(path, 'poses.pt')) 
    imgs = torch.load(os.path.join(path, 'imgs.pt'))  
    depths = torch.load(os.path.join(path, 'depths.pt'))   
    masks = torch.load(os.path.join(path, 'confidence_masks.pt'))    

    for i in range(len(poses)): 
        cam_info = {} 
        cam_info['rot'] = poses[i].detach()[:3,:3].cpu().numpy() 
        cam_info['tvec'] = poses[i].detach()[:3, 3].cpu().numpy()  
        cam_info['image'] = imgs[i]  
        cam_info['depths'] = depths[i].detach().cpu().numpy() 
        cam_info['mask'] = masks[i].cpu().numpy()   
        cam_info['height'], cam_info['width'] = imgs[i].shape[:2] 
        camera_extrinsics.append(cam_info) 

        # Print information about each image
        print(f"Image {i+1}:")
        print(f"Rotation Matrix:\n{cam_info['rot']}")
        print(f"Translation Vector: {cam_info['tvec']}")
        # print(f"Image Shape: {cam_info['image'].shape}")
        # print(f"Depth Shape: {cam_info['depths'].shape}")
        # print(f"Mask Shape: {cam_info['mask'].shape}")
        print(f"Image[0][0:3]:\n{cam_info['image'][0,0:3]}")
        print(f"Depth[0][0:3]: {cam_info['depths'][0,0:3]}")
        print(f"Mask[0][0:3]: {cam_info['mask'][0,0:3]}")
        print(f"Image Height: {cam_info['height']}, Width: {cam_info['width']}")
        print("-----------------------------------")
    
    return camera_extrinsics

def prepare_dust3r_intrinsics(path): 
    camera_intrinsics = [] 

    focals = torch.load(os.path.join(path, 'focals.pt'))  
    intrinsics = torch.load(os.path.join(path, 'intrinsics.pt'))   

    for i in range(len(focals)): 
        cam_info = {} 
        cam_info['focal'] = focals[i].item()   
        cam_info['K'] = intrinsics[i].detach().cpu().numpy()  
        camera_intrinsics.append(cam_info)

        # Print information about each image
        print(f"Camera {i+1}:")
        print(f"focal length: {cam_info['focal']}")
        print(f"K:\n{cam_info['K']}")
        print("-----------------------------------")

    return camera_intrinsics
        



