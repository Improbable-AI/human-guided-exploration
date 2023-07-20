# Copyright 2019 The ROBEL Authors.
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

"""Simulation using DeepMind Control Suite."""

import copy
import logging
from typing import Any

import dm_control.mujoco as dm_mujoco

from robel.simulation.dm_renderer import DMRenderer
from robel.simulation.sim_scene import SimScene


class DMSimScene(SimScene):
    """Encapsulates a MuJoCo robotics simulation using dm_control."""

    def _load_simulation(self, model_handle: Any) -> Any:
        """Loads the simulation from the given model handle.

        Args:
            model_handle: This can be a path to a Mujoco XML file, or an MJCF
                object.

        Returns:
            A dm_control Physics object.
        """
        if isinstance(model_handle, str):
            if model_handle.endswith('.xml'):
                sim = dm_mujoco.Physics.from_xml_path(model_handle)
            else:
                sim = dm_mujoco.Physics.from_binary_path(model_handle)
        else:
            raise NotImplementedError(model_handle)
        self._patch_mjmodel_accessors(sim.model)
        self._patch_mjdata_accessors(sim.data)
        return sim

    def _create_renderer(self, sim: Any) -> DMRenderer:
        """Creates a renderer for the given simulation."""
        return DMRenderer(sim)

    def copy_model(self) -> Any:
        """Returns a copy of the MjModel object."""
        # dm_control's MjModel defines __copy__.
        model_copy = copy.copy(self.model)
        self._patch_mjmodel_accessors(model_copy)
        return model_copy

    def save_binary(self, path: str) -> str:
        """Saves the loaded model to a binary .mjb file.

        Returns:
            The file path that the binary was saved to.
        """
        if not path.endswith('.mjb'):
            path = path + '.mjb'
        self.model.save_binary(path)
        return path

    def upload_height_field(self, hfield_id: int):
        """Uploads the height field to the rendering context."""
        if not self.sim.contexts:
            logging.warning('No rendering context; not uploading height field.')
            return
        with self.sim.contexts.gl.make_current() as ctx:
            ctx.call(self.get_mjlib().mjr_uploadHField, self.model.ptr,
                     self.sim.contexts.mujoco.ptr, hfield_id)

    def get_mjlib(self) -> Any:
        """Returns an interface to the low-level MuJoCo API."""
        return dm_mujoco.wrapper.mjbindings.mjlib

    def get_handle(self, value: Any) -> Any:
        """Returns a handle that can be passed to mjlib methods."""
        return value.ptr

    def _patch_mjmodel_accessors(self, model):
        """Adds accessors to MjModel objects to support mujoco_py API.

        This adds `*_name2id` methods to a Physics object to have API
        consistency with mujoco_py.

        TODO(michaelahn): Deprecate this in favor of dm_control's named methods.
        """
        mjlib = self.get_mjlib()

        def name2id(type_name, name):
            obj_id = mjlib.mj_name2id(model.ptr,
                                      mjlib.mju_str2Type(type_name.encode()),
                                      name.encode())
            if obj_id < 0:
                raise ValueError('No {} with name "{}" exists.'.format(
                    type_name, name))
            return obj_id

        if not hasattr(model, 'body_name2id'):
            model.body_name2id = lambda name: name2id('body', name)

        if not hasattr(model, 'geom_name2id'):
            model.geom_name2id = lambda name: name2id('geom', name)

        if not hasattr(model, 'site_name2id'):
            model.site_name2id = lambda name: name2id('site', name)

        if not hasattr(model, 'joint_name2id'):
            model.joint_name2id = lambda name: name2id('joint', name)

        if not hasattr(model, 'actuator_name2id'):
            model.actuator_name2id = lambda name: name2id('actuator', name)

        if not hasattr(model, 'camera_name2id'):
            model.camera_name2id = lambda name: name2id('camera', name)

    def _patch_mjdata_accessors(self, data):
        """Adds accessors to MjData objects to support mujoco_py API."""
        if not hasattr(data, 'body_xpos'):
            data.body_xpos = data.xpos

        if not hasattr(data, 'body_xquat'):
            data.body_xquat = data.xquat
