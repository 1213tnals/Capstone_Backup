from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from demo_run import get_3D_model_from_scene

import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

#### Functions ####
def save_ply(pcd_array:torch.Tensor, filename): 
    pcd = o3d.geometry.PointCloud()  
    pcd.points = o3d.utility.Vector3dVector(pcd_array) 
 
    o3d.io.write_point_cloud(filename, pcd) 


# 1. Get Images 
image_dir = '../images'
image_files = []

# Use os.listdir to get all files in the directory
for file in os.listdir(image_dir):
    image_files.append(os.path.join(image_dir, file)) 

print(image_files)


# 2. Make Model
# model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth"
model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
device = 'cuda'
batch_size = 1
schedule = 'linear'
lr = 0.01
niter = 300

model = load_model(model_path, device)
images = load_images(image_files, size=512)          # DUSt3R's load_images can take a list of images or a directory
pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
output = inference(pairs, model, device, batch_size=batch_size) 


# 3. Get Scene datas to share
scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

# retrieve useful values from scene:    Scene에는 image, focal, K(내부 파라미터), image_poses(외부 파라미터), point3d, mask 데이터가 들어있음
imgs = scene.imgs
focals = scene.get_focals()
poses = scene.get_im_poses()
pts3d = scene.get_pts3d()
confidence_masks = scene.get_masks() 
K = scene.get_intrinsics()
depths = scene.get_depthmaps()
conf_vals = scene.get_conf()


# 4. Save Files using .pt file format
save_dir = '../dust3r_outputs'
torch.save(imgs, f'{save_dir}/imgs.pt')  
torch.save(focals, f'{save_dir}/focals.pt')  
torch.save(poses, f'{save_dir}/poses.pt')   
torch.save(pts3d, f'{save_dir}/pts3d.pt')
torch.save(confidence_masks, f'{save_dir}/confidence_masks.pt') 
torch.save(K, f'{save_dir}/intrinsics.pt')
torch.save(depths, f'{save_dir}/depths.pt')
torch.save(conf_vals, f'{save_dir}/conf_vals.pt')

H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]      # Image Pair
img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0) 
img = np.concatenate((img0, img1), axis=1) 
# plt.imshow(img)


# 5. Make PLY file
H,W = pts3d[0].shape[:2] 
pcd_array = pts3d[0].view(H*W, 3).detach().cpu().numpy() 
save_ply(pcd_array, 'input.ply') 


# 6. Opional - Glb
silent = False
min_conf_thr = 3 
as_pointcloud = True 
mask_sky = False 
clean_depth = True 
transparent_cams = True 
cam_size = 0.05 

outfile = get_3D_model_from_scene('./', silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size) 
