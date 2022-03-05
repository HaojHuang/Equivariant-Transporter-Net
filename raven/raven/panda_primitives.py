import time
import numpy as np
import pybullet as p
import os
from transform import Transform,Rotation
import utils
from termcolor import colored


class PickPlace():
    """Pick and place primitive."""

    def __init__(self, height=0.32, speed=0.01):
        self.height, self.speed = height, speed

    def remove(self,ee):
        p.removeBody(ee.body.uid)

    def __call__(self, ee, task_name, pose0, pose1):
        """Execute pick and place primitive.
        Args:
          ee: panda effector.
          pose0: SE(3) picking pose.
          pose1: SE(3) placing pose.
        Returns:
          timeout: robot movement timed out if True.
        """
        pick_pose, place_pose = pose0, pose1
        pick_pose_ = Transform(Rotation.from_quat(pick_pose[1]), pick_pose[0])
        place_pose_ = Transform(Rotation.from_quat(place_pose[1]), place_pose[0])
        prepick_to_pick = Transform(Rotation.identity(),[0,0,self.height])
        postpick_to_pick = Transform(Rotation.identity(),[0,0,self.height])
        prepick_pose = prepick_to_pick * pick_pose_
        postpick_pose = postpick_to_pick * pick_pose_
        flip = Transform(Rotation.from_euler('y', np.pi),[0.0,0.0,0.0])
        # initialize the gripper, the pick orientation is contained in pick pose
        #TODO pre-multiply or post for flip
        ee.reset(prepick_pose*flip)
        # the gripper moving speed
        eef_step = 0.004
        vel = 0.02
        #ee.rotate_theta(np.pi/2)
        # initialized a small width to avoid colliding for some multi-objects tasks
        if task_name == 'place-red-in-green':
            ee.move(width=0.05)
        if task_name == 'stack-block-pyramid':
            ee.move(width=0.05)
        if task_name == 'packing-boxes':
            offset = Transform(Rotation.identity(),[0,0,0.01])
            pick_pose_ = offset*Transform(Rotation.from_quat(pick_pose[1]), pick_pose[0])

        if task_name == 'palletizing-boxes':
            eef_step = 0.008
            vel = 0.60

        # pick_euler = utils.quatXYZW_to_eulerXYZ(pick_pose[1])
        # theta0 = np.float(pick_euler[2])
        # theta0 = (theta0 + 2 * np.pi) % (2 * np.pi)
        # time.sleep(3)
        # ee.rotate_theta(np.pi/2)
        # time.sleep(3)
        # ee.rotate_theta(theta0)
        # move toward to pick location and abort on contact
        ee.move_tcp_xyz(pick_pose_,eef_step=eef_step,vel=vel,abort_on_contact=False)

        if ee.detect_contact():
            #self.remove(ee)
            print(colored('detect coliision during approaching to grap','red'))
            #return False
            ee.move(width=0.0)
        else:
            ee.move(width=0.0)

        if ee.check_grasp():
            #ee.activate()
            ee.grasp_object_id()
        else:
            self.remove(ee)
            print(colored('failed to grasp', 'red'))
            return False

        ee.move_tcp_xyz(postpick_pose*flip,eef_step=eef_step,vel=vel, abort_on_contact=False)
        # conduct place primitive if pick is successful
        preplace_to_place = Transform(Rotation.identity(), [0, 0, self.height])
        postplace_to_place = Transform(Rotation.identity(), [0, 0, self.height])
        preplace_pose = preplace_to_place   * place_pose_
        postplace_pose = postplace_to_place * place_pose_

        # move to the pre-place location
        ee.move_tcp_xyz(preplace_pose,eef_step=eef_step,vel=vel,abort_on_contact=False)
        #move to the pre-place pose
        place_euler = utils.quatXYZW_to_eulerXYZ(place_pose[1])
        theta = np.float(place_euler[2])
        theta = (theta+2*np.pi)%(2*np.pi)
        if theta>np.pi:
            theta = theta - 2*np.pi
        #print(colored(theta,'red'))
        ee.rotate_theta(theta)
        # #TODO discretized theta

        # for _ in range(int(self.height/0.001)):
        if task_name == 'block-insertion':
            delta = Transform(Rotation.identity(), [0, 0, 0.03])
        if task_name == 'place-red-in-green':
            delta = Transform(Rotation.identity(), [0, 0, -0.1])
        if task_name == 'align-box-corner':
            delta = Transform(Rotation.identity(), [0, 0, -0.1])
        if task_name == 'stack-block-pyramid':
            delta = Transform(Rotation.identity(), [0, 0, -0.1])
        if task_name == 'palletizing-boxes':
            delta = Transform(Rotation.identity(), [0, 0, -0.2])
        if task_name == 'packing-boxes':
            delta = Transform(Rotation.identity(), [0, 0, -0.1])



        #ee.move_tcp_xyz(delta*place_pose_,abort_on_contact=False)
        if ee.check_grasp():
            ee.grasp_object_id()
            print(colored('successful grasp', 'green'))
            ee.place_tcp_xyz(delta * place_pose_, eef_step=eef_step,vel=vel,abort_on_contact=True)
        else:
            print(colored('failed to grasp', 'red'))
            self.remove(ee)
            return False
        #time.sleep(0.1)
        # delete the constraint
        ee.release()
        # a small increase for the gripper width
        ee.move(ee.read()+0.005)
        time.sleep(0.1)
        ee.move_tcp_xyz(postplace_pose,eef_step=eef_step,vel=vel)
        self.remove(ee)
        return False