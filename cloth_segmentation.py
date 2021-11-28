import numpy as np
import open3d as o3d
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--red", type = float, default = 0.5)
parser.add_argument("--blue", type = float, default = 0.4)
parser.add_argument("--green", type = float, default = 0.4)
parser.add_argument("--source_dir", type = str, default = "./scatters")
parser.add_argument("--render", action = "store_true", default = False)
args = parser.parse_args()

# Need to consider that some cases disturbance may exist
def segment_cloth(pcd):
    color = np.array(pcd.colors)
    mask = (color[:,0] > args.red) * (color[:, 1] < args.green) * (color[:,2] < args.blue)
    points = np.asarray(pcd.points)
    truncated_pcd = o3d.geometry.PointCloud()
    truncated_pcd.points = o3d.utility.Vector3dVector(points[mask])
    truncated_pcd.colors = o3d.utility.Vector3dVector(color[mask])
    truncated_pcd.remove_statistical_outlier(nb_neighbors = 20, std_ratio = 0.04)
    return truncated_pcd

# Source direcrtory is identical to target directory
files = os.listdir(f"./pointcloud_transformed/{args.source_dir}/")
for f in files:
    filename = f"./pointcloud_transformed/{args.source_dir}/{f}"
    pcd = o3d.io.read_point_cloud(filename)
    cloth_pcd = segment_cloth(pcd)
    o3d.io.write_point_cloud(f"./pointcloud_cloth/{args.source_dir}/{f}", cloth_pcd)
    if args.render:
        o3d.visualization.draw_geometries([cloth_pcd])

