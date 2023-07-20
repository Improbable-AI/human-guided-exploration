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


class AlignBoxCorner(Task):
  """Aligning task."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_steps = 3

  def reset(self, env, poses={}, random_box_position=False, random_goal=False):
    super().reset(env)

    # Generate randomly shaped box.
    #box_size = self.get_random_size(0.05, 0.15, 0.05, 0.15, 0.01, 0.06)
    box_size = np.array([0.1, 0.1, 0.01])
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
        corner_pose = np.array([0.553125, 0.18125, 0.]), np.array([0,0,0,1])
    else:
      print("Setting predefined goal")
      corner_pose = poses['goal']

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
    box_urdf = self.fill_template(box_template, {'DIM': box_size})
    if random_box_position:
      box_pose = self.get_random_pose(env, box_size)
    else:
      box_pose = np.array([0.41249626, 0.08437618, 0.02773386]), np.array([0,0,0,1])
    box_id = env.add_object(box_urdf, box_pose)
    os.remove(box_urdf)
    self.color_random_brown(box_id)

    # Goal: box is aligned with corner (1 of 4 possible poses).
    self.goals.append(([(box_id, (2 * np.pi, None))], np.int32([[1, 1, 1, 1]]),
                       [corner_pose, pose1, pose2, pose3],
                       False, True, 'pose', None, 1))
