import open3d as o3d
import numpy as np
from PIL import Image

# PLY 파일에서 점 구름 로드
pcd = o3d.io.read_point_cloud("output_filtered_mesh_poisson.ply")

# 점 색상 정보를 추출 (RGB 값은 0~1로 정규화되어 있음)
colors = np.asarray(pcd.colors)

# 이미지 크기 설정 (예: 1024x1024)
image_size = 256
texture_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

# 점 색상 정보를 텍스처 이미지에 적용
for i, color in enumerate(colors):
    x = i % image_size
    y = i // image_size
    if y >= image_size:
        break
    texture_image[y, x] = (color * 255).astype(np.uint8)

# Pillow를 사용하여 텍스처 이미지를 PNG로 저장
texture_image = Image.fromarray(texture_image)
texture_image.save("output_texture.png")

print("텍스처 이미지가 생성되었습니다.")

