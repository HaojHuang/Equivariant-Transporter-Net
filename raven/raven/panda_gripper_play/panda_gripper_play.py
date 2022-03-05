from pybullet_utils import bullet_client
import time
import numpy as np
import pybullet
from transform import Transform,Rotation
from pathlib import Path
from wrap_pybullet_utils import Body,Link,Joint,Constraint
import numpy as np

assert pybullet.isNumpyEnabled()  # better performance of transforming sensor image(c++ based) to numpy

from perception import create_tsdf,render_images
import open3d as o3d
from sim import Sim,Grasp


sim = Sim()
sim.reset()
#sim.gripper.reset(Transform(Rotation.identity(),np.r_[0,0,0.5]),opening_width=0.8)
# load object
sim.load_object_pile_mode()
sim.load_object_packed_mode()

# 2 depth images in pybullet
depth_imgs, extrinsics = render_images(sim, 2)
# create voxesl grid
tsdf = create_tsdf(sim.size, 80, depth_imgs, sim.camera.intrinsic, extrinsics)
pcd = tsdf.get_cloud()
bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
pcd = pcd.crop(bounding_box)
#show the piont cloud
o3d.visualization.draw_geometries([pcd])
for i in range(10):
    grasp_candiadate = Grasp(pose=Transform(Rotation.identity(),np.r_[0.15,0.15,0.4]),width=0.4)
    label=sim.execute_grasp(grasp_candiadate,remove=True,allow_contact=False)
    print(label)
sim.wait()

