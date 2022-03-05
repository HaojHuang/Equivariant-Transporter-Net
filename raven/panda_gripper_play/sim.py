from pathlib import Path
import time
import numpy as np
import pybullet
import enum
from perception import *
import wrap_pybullet_utils
from workspace_lines import  workspace_lines
from transform import Rotation,Transform


class Label(enum.IntEnum):
    FAILURE = 0  # grasp execution failed due to collision or slippage
    SUCCESS = 1  # object was successfully removed

class Grasp(object):
    """Grasp parameterized as pose of a 2-finger robot hand.
    TODO(mbreyer): clarify definition of grasp frame
    """
    def __init__(self, pose, width):
        self.pose = pose
        self.width = width

class Gripper(object):
    """Simulated Panda hand."""

    def __init__(self, world):
        self.world = world
        self.urdf_path = Path("panda/hand.urdf")
        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.022])
        #self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.0])
        self.T_tcp_body = self.T_body_tcp.inverse()

    def reset(self, T_world_tcp,opening_width=0.08):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)
        # sets the position of the COM, not URDF link
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.update_tcp_constraint(T_world_tcp)
        # constraint to keep fingers centered
        self.world.add_constraint(
            self.body,
            self.body.links["panda_leftfinger"],
            self.body,
            self.body.links["panda_rightfinger"],
            pybullet.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        ).change(gearRatio=-1, erp=0.1, maxForce=50)

        self.joint1 = self.body.joints["panda_finger_joint1"]
        self.joint1.set_position(0.5 * opening_width, kinematics=True)
        self.joint2 = self.body.joints["panda_finger_joint2"]
        self.joint2.set_position(0.5 * opening_width, kinematics=True)

    def update_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )

    def set_tcp(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self, threshold=5):
        #time.sleep(1)
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def move(self, width):
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read(self):
        width = self.joint1.get_position() + self.joint2.get_position()
        return width


class Sim(object):
    def __init__(self,gui=True,seed=None):
        self.gui = gui
        self.rng = np.random.RandomState(seed) if seed else np.random
        # built the pybullet world
        self.world = wrap_pybullet_utils.BtWorld(self.gui)
        # initialize the gripper
        self.gripper = Gripper(self.world)
        intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)
        # the gripper.finger_depth = 0.05
        self.size = 8 * self.gripper.finger_depth

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies())

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def draw_workspace(self):
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self,height=0.05):
        urdf = 'setup/plane.urdf'
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        self.world.load_urdf(urdf, pose, scale=0.6)
        # define valid volume for sampling grasps (the volume for the workspace)
        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        self.lower = np.r_[lx, ly, lz]
        self.upper = np.r_[ux, uy, uz]

    def load_object_pile_mode(self,urdf = 'setup/chips_can.urdf',table_height=0.05):
        # we load object into the scene by droping it with random pose, and thus get a pile scene
        rotation = Rotation.random(random_state=self.rng)
        xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
        pose = Transform(rotation, np.r_[xy, table_height + 0.2])
        scale = self.rng.uniform(0.8, 1.0)
        self.world.load_urdf(urdf, pose, scale=scale*0.5)
        self.wait_for_objects_to_rest(timeout=1.0)

    def load_object_packed_mode(self,urdf = 'setup/chips_can.urdf',table_height=0.05):
        # load object into the scene by dropping it with canonical pose
        #pre_num = self.num_objects
        attempts = 0
        max_attempts = 12
        while attempts < max_attempts:
            self.save_state()
            x = self.rng.uniform(0.08, 0.22)
            y = self.rng.uniform(0.08, 0.22)
            z = 1.0
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            pose = Transform(rotation, np.r_[x, y, z])
            scale = self.rng.uniform(0.7, 0.9)
            body = self.world.load_urdf(urdf, pose, scale=scale*0.5)
            # query the axis aligned bounding box (in world space)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.world.step()
            # we don't want the new dropped object collide with others
            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
                break
            attempts += 1




    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            #if np.any(xyz < 0.0) or np.any(xyz > self.size):
            if np.any(xyz < 0.0) or np.any(xyz[:2] > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object


    def reset(self):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )
        self.place_table()




    def wait(self,):
        while True:
            self.world.step()

    # perception part
    def acquire_tsdf(self, n, N=None):
        """Render synthetic depth images from n viewpoints and integrate into a TSDF.
        If N is None, the n viewpoints are equally distributed on circular trajectory.
        If N is given, the first n viewpoints on a circular trajectory consisting of N points are rendered.
        """
        tsdf = TSDFVolume(self.size, 40)

        #high resolution tsdf
        high_res_tsdf = TSDFVolume(self.size, 120)

        origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, 0])
        r = 2.0 * self.size
        theta = np.pi / 6.0

        N = N if N else n
        phi_list = 2.0 * np.pi * np.arange(n) / N
        extrinsics = [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]
        timing = 0.0
        for extrinsic in extrinsics:
            depth_img = self.camera.render(extrinsic)[1]
            tic = time.time()
            tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
            timing += time.time() - tic
            high_res_tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
        return tsdf, high_res_tsdf.get_cloud(), timing

    def execute_grasp(self, grasp, remove=True, allow_contact=False):
        # --grasp is the target containing pose and width
        # -- flag to control whether allow collision between pre-target and target
        # -- remove whether remove the objec from the scene after succesful grasp
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        # approach along z-axis of the gripper
        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        # move the gripper to pregrasp pose and detect the collision
        self.gripper.reset(T_world_pregrasp)
        if self.gripper.detect_contact():
            result = Label.FAILURE, self.gripper.max_opening_width
        else:
            #move the gripper to the target pose and detect collision
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, self.gripper.max_opening_width
            else:
                self.gripper.move(0.0) # shrink the gripper
                #lift the gripper up along z-axis of the world frame or z-axis of the gripper frame
                self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                if self.check_success(self.gripper):
                    result = Label.SUCCESS, self.gripper.read()
                    if remove:
                        contacts = self.world.get_contacts(self.gripper.body)
                        self.world.remove_body(contacts[0].bodyB)
                else:
                    result = Label.FAILURE, self.gripper.max_opening_width

        self.world.remove_body(self.gripper.body)

        if remove:
            self.remove_and_wait()
        return result

    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts = self.world.get_contacts(gripper.body)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res




# from perception import create_tsdf,render_images
# import open3d as o3d
# sim = Sim()
# #sim.wait()
# sim.reset()
# sim.gripper.reset(Transform(Rotation.identity(),np.r_[0,0,0.5]),opening_width=0.8)
# sim.load_object_pile_mode()
# sim.load_object_packed_mode()
# depth_imgs, extrinsics = render_images(sim, 2)
# tsdf = create_tsdf(sim.size, 160, depth_imgs, sim.camera.intrinsic, extrinsics)
# pcd = tsdf.get_cloud()
# bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
# pcd = pcd.crop(bounding_box)
# o3d.visualization.draw_geometries([pcd])
# for i in range(10):
#     grasp_candiadate = Grasp(pose=Transform(Rotation.identity(),np.r_[0.1,0.1,0.2]),width=0.4)
#     label=sim.execute_grasp(grasp_candiadate,remove=True,allow_contact=False)
#     print(label)
# sim.wait()