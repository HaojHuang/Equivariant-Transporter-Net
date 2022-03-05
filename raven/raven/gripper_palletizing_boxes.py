# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Palletizing Task."""

import os
import numpy as np
from gripper_task import Task
import utils
import pybullet as p
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
class PalletizingBoxes(Task):
  """Palletizing Task."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_steps = 30
    self.task_name = 'palletizing-boxes'
  def reset(self, env):
    super().reset(env)

    # Add pallet.
    zone_size = (0.3, 0.25, 0.25)
    zone_urdf = 'pallet/pallet.urdf'
    rotation = utils.eulerXYZ_to_quatXYZW((0, 0, 0))
    zone_pose = ((0.5, 0.25, 0.02), rotation)
    env.add_object(zone_urdf, zone_pose, 'fixed')

    # Add stack of boxes on pallet.
    margin = 0.035
    margin_x = 0.02
    margin_y = 0.035
    object_ids = []
    object_points = {}
    stack_size = (0.19, 0.19, 0.19)
    box_template = 'box/box-template.urdf'
    stack_dim = np.int32([2, 3, 3])
    # stack_dim = np.random.randint(low=2, high=4, size=3)
    box_size = (stack_size - (stack_dim - 1) * margin) / stack_dim
    box_size[0] = 0.085
    for z in range(stack_dim[2]):
      # Transpose every layer.
      stack_dim[0], stack_dim[1] = stack_dim[1], stack_dim[0]
      box_size[0], box_size[1] = box_size[1], box_size[0]
      margin_y, margin_x = margin_x, margin_y
      for y in range(stack_dim[1]):
        for x in range(stack_dim[0]):
          position = list((x + 0.5, y + 0.5, z + 0.5) * box_size)
          position[0] += x * margin_x - stack_size[0] / 2
          position[1] += y * margin_y - stack_size[1] / 2
          position[2] += z * margin + 0.03
          pose = (position, (0, 0, 0, 1))
          pose = utils.multiply(zone_pose, pose)
          urdf = self.fill_template(box_template, {'DIM': box_size})
          box_id = env.add_object(urdf, pose)
          os.remove(urdf)
          object_ids.append((box_id, (0, None)))
          self.color_random_brown(box_id)
          object_points[box_id] = self.get_object_points(box_id)

    # Randomly select top box on pallet and save ground truth pose.
    targets = []
    self.steps = []
    boxes = [i[0] for i in object_ids]
    while boxes:
      _, height, object_mask = self.get_true_image(env)
      top = np.argwhere(height > (np.max(height) - 0.03))
      rpixel = top[int(np.floor(np.random.random() * len(top)))]  # y, x
      box_id = int(object_mask[rpixel[0], rpixel[1]])
      if box_id in boxes:
        position, rotation = p.getBasePositionAndOrientation(box_id)
        rposition = np.float32(position) + np.float32([0, -10, 0])
        p.resetBasePositionAndOrientation(box_id, rposition, rotation)
        self.steps.append(box_id)
        targets.append((position, rotation))
        boxes.remove(box_id)
    self.steps.reverse()  # Time-reversed depalletizing.

    self.goals.append((
        object_ids, np.eye(len(object_ids)), targets, False, True,
        'zone', (object_points, [(zone_pose, zone_size)]), 1))

    self.spawn_box()

  def reward(self):
    reward, info = super().reward()
    self.spawn_box()
    return reward, info

  def spawn_box(self):
    """Palletizing: spawn another box in the workspace if it is empty."""
    workspace_empty = True
    if self.goals:
      for obj, _ in self.goals[0][0]:
        obj_pose = p.getBasePositionAndOrientation(obj)
        workspace_empty = workspace_empty and ((obj_pose[0][1] < -0.5) or
                                               (obj_pose[0][1] > 0))
      if not self.steps:
        self.goals = []
        print('Palletized boxes toppled. Terminating episode.')
        return

      if workspace_empty:
        obj = self.steps[0]
        theta = np.random.random() * 2 * np.pi
        rotation = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        p.resetBasePositionAndOrientation(obj, [0.5, -0.25, 0.1], rotation)
        self.steps.pop(0)

    # Wait until spawned box settles.
    for _ in range(480):
      p.stepSimulation()