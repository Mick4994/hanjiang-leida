import open3d as o3d
import numpy as np
import os

all_pcd_file = os.listdir('3D-models/terra_pcd/')
all_pcd_file = ['3D-models/terra_pcd/' + i for i in all_pcd_file if i.endswith('.pcd')]
pcd_list = [o3d.io.read_point_cloud(pcd_file) for pcd_file in all_pcd_file]

xyz_list = []
color_list = []

for i in pcd_list:
    xyz_list += np.asarray(i.points).tolist()
    color_list += np.asarray(i.colors).tolist()
xyz = np.array(xyz_list)
colors = np.array(color_list)//10*10
print(colors)

# sztu_pcd = o3d.geometry.PointCloud()
# sztu_pcd.points = o3d.utility.Vector3dVector(xyz)
# sztu_pcd.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw([sztu_pcd])


