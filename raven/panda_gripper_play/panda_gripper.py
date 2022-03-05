
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import numpy as np
import pybullet
from transform import Rotation,Transform
import pybullet as p



class Gripper(object):
    """Simulated Panda hand."""

    def __init__(self,path,obj_ids=None):
        self.urdf_path = path
        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.022])
        #self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.0])
        self.T_tcp_body = self.T_body_tcp.inverse()
        self.dt = 1.0 / 240.0
        p.setTimeStep(self.dt)
        self.activated = False
        self.contact_constraint = None
        self.obj_ids = None
        if obj_ids is not None:
            self.obj_ids = obj_ids
            #print('======================',self.obj_ids)
    def step(self):
        p.stepSimulation()

    def load_urdf(self, urdf_path, pose, scale=1.0):
        body = Body.from_urdf(urdf_path, pose, scale)
        return body

    def add_constraint(self,*argv, **kwargs):
        """See `Constraint` below."""
        constraint = Constraint(*argv, **kwargs)
        return constraint

    def reset(self, T_world_tcp,opening_width=0.08):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)
        # sets the position of the COM, not URDF link
        self.constraint = self.add_constraint(
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
        self.add_constraint(
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


    def move_tcp_xyz(self, target, eef_step=0.004, vel=0.20, abort_on_contact=True):
        # eef_step=0.004, vel=0.10,
        #print(eef_step,vel)
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        if n_steps==0:
            n_steps=1 # avoid divide by zero
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.dt)):
                self.step()
            if abort_on_contact and self.detect_contact():
            #if self.check_object_contact():
                return

    def place_tcp_xyz(self, target, eef_step=0.004, vel=0.20, abort_on_contact=True):
        # eef_step=0.004, vel=0.10,
        # Todo by haojie add place position control for panda gripper
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        if n_steps==0:
            n_steps=1 # avoid divide by zero
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.dt)):
                self.step()
                if abort_on_contact:
                    if self.check_object_contact() or self.check_gripper_collide():
                        return


    def rotate_theta(self,theta, eef_step=0.05, vel=0.80, axis='z'):
        #eef_step=0.05, vel=0.40,
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp
        pre_position = T_world_tcp.translation
        diff = theta
        n_step = int(abs(theta)/eef_step)
        if n_step==0:
            n_step=1 # avoid divide by zero
        dist_step = diff/n_step
        dur_step = abs(dist_step) /vel
        for _ in range(n_step):
            #T_world_tcp = Transform(Rotation.from_euler(axis,dist_step),[0.0,0.0,0.0]) * T_world_tcp
            T_world_tcp = T_world_tcp*Transform(Rotation.from_euler(axis, -dist_step), [0.0, 0.0, 0.0])
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step/self.dt)):
                self.step()

    def get_contacts(self, bodyA):
        # only report contact points that involve body A
        contact_points = p.getContactPoints(bodyA.uid)
        # record each contact point
        contacts = []
        for point in contact_points:
            contact = Contact(
                bodyA=point[1],
                bodyB=point[2],
                point=point[5],
                normal=point[7], # -- normal is contact normal on B, pointing towards A
                depth=point[8],
                force=point[9],
            )
            contacts.append(contact)
        return contacts

    def detect_contact(self, threshold=5):
        # time.sleep(1)
        if self.get_contacts(self.body):
            return True
        else:
            return False

    def move(self, width):
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.dt)):
            self.step()

    def read(self):
        width = self.joint1.get_position() + self.joint2.get_position()
        return width

    def activate(self):
        if not self.activated:
            points = p.getContactPoints(bodyA=self.body.uid)
            if points:
                #print(len(points))
                for point in points:
                    #print(len(point))
                    obj_id, finger, contact_link = point[2], point[3],point[4]
                    if finger==0:
                        break

                if finger==0:#obj_id in self.obj_ids['rigid']:
                    body_pose = p.getLinkState(self.body.uid, 0)
                    obj_pose = p.getBasePositionAndOrientation(obj_id)
                    world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                    obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                       world_to_body[1],
                                                       obj_pose[0], obj_pose[1])

                    self.contact_constraint = p.createConstraint(
                        parentBodyUniqueId=self.body.uid,
                        parentLinkIndex=0,
                        childBodyUniqueId=obj_id,
                        childLinkIndex=contact_link,
                        jointType=p.JOINT_FIXED,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=obj_to_body[0],
                        parentFrameOrientation=obj_to_body[1],
                        childFramePosition=(0, 0, 0),
                        childFrameOrientation=(0, 0, 0))
                self.activated = True

    def release(self):
        if self.activated:
            self.activated = False
            # Release gripped rigid object (if any).
            if self.contact_constraint is not None:
                try:
                    p.removeConstraint(self.contact_constraint)
                    self.contact_constraint = None
                except:  # pylint: disable=bare-except
                    pass
    def check_grasp(self):
        contacts = self.get_contacts(self.body)
        res = len(contacts) > 0 and self.read() > 0.1 * self.max_opening_width
        return res

    def check_object_contact(self):
        #TODO BY HAOJIE
        # detect whether the gripper grasps an object
        if self.check_grasp():
            contacts = self.get_contacts(self.body)
            #print(len(contacts))
            #just need one contact
            for contact in contacts:
                #contact = contacts[0]
                # get rid body
                body = contact.bodyB
                if self.obj_ids is None or body in self.obj_ids['rigid']:
                    # contact point between rigid body and others
                    points = p.getContactPoints(bodyA=body)
                    # remove contacts between gripper and object
                    points = [point for point in points if point[2] != self.body.uid]
                    if points:
                        return True
            return False
        else:# if there is no grasp just pass
            pass

    def grasp_object_id(self):
        if self.check_grasp():
            contacts = self.get_contacts(self.body)
            # print(len(contacts))
            # just need one contact
            for contact in contacts:
                # contact = contacts[0]
                # get rid body
                body = contact.bodyB
                if body in self.obj_ids['rigid']:
                    self.grasped_id = body
        else:
            self.grasped_id = None


    def check_gripper_collide(self):
        if self.grasped_id:
            contacts = self.get_contacts(self.body)
            #just need one contact
            for contact in contacts:
                #contact = contacts[0]
                # get rid body
                body = contact.bodyB
                if body!=self.grasped_id:
                    return True
            return False
        else:# if there is no grasp just pass
            pass





class Constraint(object):
    """Interface to a constraint in PyBullet.
    Attributes:
        uid: The unique id of the constraint within the physics server.
    """

    def __init__(
        self,
        parent,
        parent_link,
        child,
        child_link,
        joint_type,
        joint_axis,
        parent_frame,
        child_frame,
    ):
        """
        Create a new constraint between links of bodies.
        Args:
            parent:
            parent_link: None for the base.
            child: None for a fixed frame in world coordinates.
        """
        self.p = p
        parent_body_uid = parent.uid
        parent_link_index = parent_link.link_index if parent_link else -1
        child_body_uid = child.uid if child else -1
        child_link_index = child_link.link_index if child_link else -1

        # constraint.uid
        self.uid = self.p.createConstraint(
            parentBodyUniqueId=parent_body_uid,
            parentLinkIndex=parent_link_index,
            childBodyUniqueId=child_body_uid,
            childLinkIndex=child_link_index,
            jointType=joint_type,
            jointAxis=joint_axis,
            parentFramePosition=parent_frame.translation,
            parentFrameOrientation=parent_frame.rotation.as_quat(),
            childFramePosition=child_frame.translation,
            childFrameOrientation=child_frame.rotation.as_quat(),
        )

    def change(self, **kwargs):
        self.p.changeConstraint(self.uid, **kwargs)

class Body(object):
    """Interface to a multibody simulated in PyBullet.
    Attributes:
        uid: The unique id of the body within the physics server.
        name: The name of the body.
        joints: A dict mapping joint names to Joint objects.
        links: A dict mapping link names to Link objects.
    """

    def __init__(self, body_uid):
        self.p = p
        self.uid = body_uid
        self.name = self.p.getBodyInfo(self.uid)[1].decode("utf-8")
        # a body contains several links and joints
        self.joints, self.links = {}, {}
        #TODO by haojie, add revolute joints {}
        for i in range(self.p.getNumJoints(self.uid)):
            joint_info = self.p.getJointInfo(self.uid, i)
            joint_name = joint_info[1].decode("utf8")
            self.joints[joint_name] = Joint( self.uid, i)
            link_name = joint_info[12].decode("utf8")
            self.links[link_name] = Link(self.uid, i)

    @classmethod
    def from_urdf(cls, urdf_path, pose, scale=1.):

        # the pose is defined as a transformation class has rotation and translation (check transform.py for details)
        body_uid = p.loadURDF(
            str(urdf_path),
            pose.translation,
            pose.rotation.as_quat(),
            globalScaling=scale,
        )
        return cls(body_uid)

    def get_pose(self):
        pos, ori = self.p.getBasePositionAndOrientation(self.uid)
        return Transform(Rotation.from_quat(ori), np.asarray(pos))

    def set_pose(self, pose):
        self.p.resetBasePositionAndOrientation(
            self.uid, pose.translation, pose.rotation.as_quat()
        )

    def get_velocity(self):
        linear, angular = self.p.getBaseVelocity(self.uid)
        return linear, angular


class Link(object):
    """Interface to a link simulated in Pybullet.
    Attributes:
        link_index: The index of the joint.
    """

    def __init__(self, body_uid, link_index):
        self.p = p
        self.body_uid = body_uid
        self.link_index = link_index

    def get_pose(self):
        link_state = self.p.getLinkState(self.body_uid, self.link_index)
        pos, ori = link_state[0], link_state[1]
        return Transform(Rotation.from_quat(ori), pos)

    def get_position(self):
        link_state = self.p.getLinkState(self.body_uid, self.link_index)
        pos = link_state[0]
        return pos


class Joint(object):
    """Interface to a joint simulated in PyBullet.
    Attributes:
        joint_index: The index of the joint.
        lower_limit: Lower position limit of the joint.
        upper_limit: Upper position limit of the joint.
        effort: The maximum joint effort.
    """

    def __init__(self, body_uid, joint_index):
        self.p = p
        self.body_uid = body_uid
        self.joint_index = joint_index

        joint_info = self.p.getJointInfo(body_uid, joint_index)
        self.lower_limit = joint_info[8]
        self.upper_limit = joint_info[9]
        self.effort = joint_info[10]

    def get_position(self):
        joint_state = self.p.getJointState(self.body_uid, self.joint_index)
        return joint_state[0]

    def set_position(self, position, kinematics=False):
        if kinematics:
            # directly reset, no velocity involved
            self.p.resetJointState(self.body_uid, self.joint_index, position)

        self.p.setJointMotorControl2(
            self.body_uid,
            self.joint_index,
            pybullet.POSITION_CONTROL,
            targetPosition=position,
            force=self.effort,
        )

class Contact(object):
    """Contact point between two multibodies.
    Attributes:
        point: Contact point.
        normal: Normal vector from ... to ...
        depth: Penetration depth
        force: Contact force acting on body ...
    """

    def __init__(self, bodyA, bodyB, point, normal, depth, force):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.point = point
        self.normal = normal
        self.depth = depth
        self.force = force