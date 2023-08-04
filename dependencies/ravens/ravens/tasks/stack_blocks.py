# coding=utf-8
# Copyright 2022 The Ravens Authors.
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

"""Aligning task."""

import os

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils
import pybullet as p



class StackBlocks(Task):
  """Aligning task."""

  def __init__(self, num_blocks=2, *args, **kwargs):
    super().__init__(*args, **kwargs)
    #self.max_steps = 3
    self.num_blocks = num_blocks

 
    self.num_boxes = num_blocks


  def color_random(self, obj):
        colors = [
        utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
        utils.COLORS['yellow'], utils.COLORS['orange'], utils.COLORS['red']
        ]

        idx = np.random.randint(len(colors))
        p.changeVisualShape(obj, -1, rgbaColor=colors[idx])

  def color_box(self, obj, block_idx=0):
        colors = [
        np.float32([250, 253, 15, 255])/255,
        np.float32([46, 139, 87, 255])/255,
        np.float32([0, 150, 255, 255])/255,
        np.float32([220, 20, 60, 255])/255,
        ]

        p.changeVisualShape(obj, -1, rgbaColor=colors[block_idx])

  def reset(self, env, poses={}, random_box_position=False, random_goal=False, reset_to_goal=False):
    super().reset(env)

    # Generate randomly shaped box.
    #box_size = self.get_random_size(0.05, 0.15, 0.05, 0.15, 0.01, 0.06)
    box_size = np.array([0.05, 0.05, 0.05])
    # Add corner.
   
    dimx = (box_size[0] / 2 - 0.025 + 0.0025, box_size[0] / 2 + 0.0025)
    dimy = (box_size[1] / 2 + 0.0025, box_size[1] / 2 - 0.025 + 0.0025)
    corner_template = 'corner/corner-template.urdf'
    replace = {'DIMX': dimx, 'DIMY': dimy}
    corner_urdf = self.fill_template(corner_template, replace)
    corner_size = (box_size[0], box_size[1], 0)
    if not 'goal' in poses:
      if random_goal:
        corner_pose = self.get_random_pose(env, corner_size)
      else:
        corner_pose = np.array([0.653125, 0.18125, 0.]), np.array([0,0,0,1])
    else:
      print("Setting predefined goal")
      corner_pose = poses['goal']
    if self.num_blocks != 4:
      env.add_object(corner_urdf, corner_pose, 'fixed')
    os.remove(corner_urdf)

    # Add possible placing poses.
    theta = utils.quatXYZW_to_eulerXYZ(corner_pose[1])[2]
    fip_rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta + np.pi))
    pose1 = (corner_pose[0], fip_rot)
    alt_x = (box_size[0] / 2) - (box_size[1] / 2)
    alt_y = (box_size[1] / 2) - (box_size[0] / 2)
    alt_pos = (alt_x, alt_y, 0)
    alt_rot0 = utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2))
    alt_rot1 = utils.eulerXYZ_to_quatXYZW((0, 0, 3 * np.pi / 2))
    pose2 = utils.multiply(corner_pose, (alt_pos, alt_rot0))
    pose3 = utils.multiply(corner_pose, (alt_pos, alt_rot1))

    # Add box.
    box_template = 'box/box-template.urdf'
    if reset_to_goal:
      box_poses = np.array([[0.57, 0.14, 0.17],[0.57, 0.14, 0.15],[0.57, 0.08, 0.001], [0.57, 0.2, 0.001]])
    else:
      box_poses = np.array([[0.61249626, -0.2, 0.02773386],[0.41249626, 0.08437618, 0.02773386], [0.5, 0.2, 0.02773386], [0.41249626, -0.2, 0.02773386]])
    
    if self.num_blocks == 4:
        box_templates = [box_template, box_template, box_template]
        box_sizes = [[0.002, 0.002, 0.002],[0.006, 0.002, 0.002], [0.002, 0.002, 0.003], [0.002, 0.002, 0.003]]
        obj_shapes = [10, 8, 2, 11] # 8, 2, 11 #5, 6 # 2: triangle 6 11:circle 8 rectangle 12 G
        for i in range(self.num_boxes):
          template = 'kitting/object-template.urdf'
          shape = os.path.join(self.assets_root, 'kitting',
                              f'{obj_shapes[i]:02d}.obj')
          scale = box_sizes[i] #[0.003, 0.003, 0.0001]  # .0005
          replace = {'FNAME': (shape,), 'SCALE': scale, 'COLOR': (0.2, 0.2, 0.2)}
          box_urdf = self.fill_template(template, replace)
          #box_urdf = self.fill_template(box_templates[i], {'DIM': box_sizes[i]})
          box_pose = box_poses[i], np.array([0,0,0.707,0.707])#np.array([0,0,-0.383,0.924])
          box_id = env.add_object(box_urdf, box_pose)
          os.remove(box_urdf)
          self.color_box(box_id, i)
    else:
      for i in range(self.num_boxes):
          box_urdf = self.fill_template(box_template, {'DIM': box_size})
          if random_box_position:
              box_pose = self.get_random_pose(env, box_size)
          else:
              box_pose = box_poses[i], np.array([0,0,0,1])
          box_id = env.add_object(box_urdf, box_pose)
          os.remove(box_urdf)
          self.color_box(box_id, i)

    # Goal: box is aligned with corner (1 of 4 possible poses).
    self.goals.append(([(box_id, (2 * np.pi, None))], np.int32([[1, 1, 1, 1]]),
                      [corner_pose, pose1, pose2, pose3],
                      False, True, 'pose', None, 1))