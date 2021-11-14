import numpy as np
import open3d as o3d
import os

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--source_dir", type=str, default="scatters")
parser.add_argument("--target_dir", type=str, default="scatters")
parser.add_argument("--extrinsic",type=str, default = "extrinsic")
parser.add_argument("--render",action="store_true", default = False)
args = parser.parse_args()

source_files = os.listdir(f"./pointcloud_raw/{args.source_dir}")
pcds = []
for file in source_files:
    filename = f"./pointcloud_raw/{args.source_dir}/{file}"
    pcd = o3d.io.read_point_cloud(filename)
    color = np.asarray(pcd.colors)
    color[:,[0,2]] = color[:,[2,0]]
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcds.append(pcd)

extrinsics = np.load(f"./extrinsic/{args.extrinsic}.npz")
R, T = extrinsics["R"], extrinsics["T"]

os.makedirs(f"./pointcloud_transformed/{args.target_dir}",exist_ok=True)
for pcd in pcds:
    pcd.translate(T)
    pcd.rotate(R, center=(0,0,0))
    o3d.io.write_point_cloud(f"./pointcloud_transformed/{args.target_dir}/{source_files.pop(0)}", pcd)

if args.render:
    o3d.visualization.draw_geometries([pcds[-1]])
