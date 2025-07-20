'''
Created by Basile Van Hoorick for GCD, 2024.
Based on TCOW, 2023.
Single Kubric simulator instance.
Adapted from movi_def_worker.py.
'''

import os, certifi  # noqa

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()
import sys  # noqa

sys.path.insert(0, os.path.join(os.getcwd(), 'kubric/'))  # noqa
sys.path.insert(0, os.path.join(os.getcwd()))  # noqa

# Library imports.
import argparse
import collections
import collections.abc
import colorsys
import copy
import datetime
import glob
import itertools
import json
import math
import multiprocessing as mp
import pathlib
import pickle
import platform
import random
import shutil
import sys
import tempfile
import time
import warnings
from collections import defaultdict

import cv2
import fire
import imageio
import joblib
import lovely_numpy
import lovely_tensors
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich
import rich.console
import rich.logging
import rich.progress
import scipy
import tqdm
import tqdm.rich
from einops import rearrange, repeat
from lovely_numpy import lo
from rich import print
from tqdm import TqdmExperimentalWarning
import trimesh
from scipy.spatial.transform import Rotation as R
import shutil
import lzma
from pathlib import Path
import os, sys

# Internal imports.
from kubric_constants import *  # noqa

np.set_printoptions(precision=3, suppress=True)

CONTAINER_CARRIER_SPAWN_REGION = [(-4, -4, 0), (4, 4, 4)]
STATIC_SPAWN_REGION = [(-7, -7, 0), (7, 7, 7)]
DYNAMIC_SPAWN_REGION = [(-5, -5, 1), (5, 5, 6)]
VELOCITY_RANGE = [(-4, -4, -1), (4, 4, 1)]


class MyKubricSimulatorRenderer:
    '''
    This class is capable of generating Kubric scenes in a customizable way.
    It can be used both for offline and online simulation and rendering.
    '''

    def __init__(self,
                 frame_width=256,
                 frame_height=192,
                 num_frames=24,
                 frame_rate=12,
                 motion_blur=True,
                 render_samples_per_pixel=32,
                 split_backgrounds=False,
                 split_objects=False,
                 render_use_gpu=False,
                 render_cpu_threads=-1,
                 scratch_dir=None,
                 mass_est_fp=None,
                 dome_friction_range=[0.3, 0.4],
                 dome_restit_range=[0.6, 0.7],
                 object_friction_range=[0.4, 0.5],
                 object_restit_range=[0.6, 0.7],
                 max_camera_speed=8.0,
                 mass_scaling_law=3.0):
        '''
        Initializes the context for Kubric and stores some data distribution parameters.
        '''
        # WARNING / NOTE: We CANNOT import bpy outside of the actual thread / process using it!
        # Otherwise, a lot of problems / crashes will occur that do not seem to be easily fixable.
        # Therefore, kubric stuff also has to be placed here because it itself imports bpy.
        import bpy
        import kubric as kb
        import kubric.renderer
        import kubric.simulator

        # Keep the bpy and Kubric imports alive throughout the lifetime of this object instance.
        # Feels weird but this seems to be the only way to use their functionality in other methods.
        self.bpy_module = bpy  #blender python API
        self.kb_module = kb  #Kubric simolator and renderer

        self.frame_width = frame_width  # output image width (pixels)
        self.frame_height = frame_height  # output image height (pixels)
        self.num_frames = num_frames  # frames per video clip
        self.frame_rate = frame_rate  # frames-per-second used by simulator & renderer
        self.motion_blur = motion_blur  # enable motion-blur (randomized strength if True)
        self.render_samples_per_pixel = render_samples_per_pixel  # Cycles path-tracing samples per pixel
        self.split_backgrounds = split_backgrounds  # train/test split HDRI backgrounds separately
        self.split_objects = split_objects  # train/test split object asset pool separately
        self.render_use_gpu = False  #render_use_gpu  # let Blender use GPU if available
        self.render_cpu_threads = render_cpu_threads  # fixed CPU thread count for rendering (-1 → auto)
        self.mass_est_fp = mass_est_fp  # CSV file with per-asset mass priors
        self.dome_friction_range = dome_friction_range  # rng for ground (dome) friction coeff
        self.dome_restit_range = dome_restit_range  # rng for ground restitution (bounciness)
        self.object_friction_range = object_friction_range  # rng for object surface friction
        self.object_restit_range = object_restit_range  # rng for object restitution
        self.max_camera_speed = max_camera_speed  # upper bound on random camera linear speed
        self.mass_scaling_law = mass_scaling_law  # exponent to scale mass ∝ scale^power

        # Initialize scratch directory.
        if scratch_dir is None:
            scratch_dir = tempfile.mkdtemp()
        else:
            # scratch_dir = os.path.join(scratch_dir, str(np.random.randint(1000000000, 9999999999)))
            os.makedirs(scratch_dir, exist_ok=True)
            print(f'Using scratch directory: {scratch_dir}.')

        # Initialize variables.
        self.scene = None
        self.scratch_dir = scratch_dir
        self.simulator = None
        self.renderer = None

        # Prepare asset sources; same as MOVi-D/E/F.
        self.gso_source = kb.AssetSource.from_manifest(  #All google scanned objects
            'gs://kubric-public/assets/GSO/GSO.json')
        self.hdri_source = kb.AssetSource.from_manifest(  #HDRI objects
            'gs://kubric-public/assets/HDRI_haven/HDRI_haven.json')
        self.kubasic_source = kb.AssetSource.from_manifest(
            'gs://kubric-public/assets/KuBasic/KuBasic.json')
        self.gso_shoe_ids = self._asset_ids_from_contains(GSO_SHOE_CONTAINS)
        self.gso_box_ids = self._asset_ids_from_contains(GSO_BOX_CONTAINS)

        # Set rendering options.
        if self.render_cpu_threads >= 1:
            self.bpy_module.context.scene.render.threads_mode = 'FIXED'
            self.bpy_module.context.scene.render.threads = render_cpu_threads
        os.environ['KUBRIC_USE_GPU'] = '1' if render_use_gpu else '0'
        print(f'KUBRIC_USE_GPU: {render_use_gpu}')

        # Load mass knowledge.
        # It is recommended to use mass_min_max_dict, which maps asset IDs to practical mass ranges.
        if self.mass_est_fp is not None:
            mass_est_list = pd.read_csv(self.mass_est_fp,
                                        header=None,
                                        names=['id', 'samples'])
            self.mass_samples_dict = {
                id: np.fromstring(samples[1:-1], dtype=np.float32, sep=' ')
                for (id, samples) in mass_est_list.values
            }
            self.mass_min_max_dict = {
                id: (samples.mean() * 0.5, samples.mean() * 1.5)
                for (id, samples) in self.mass_samples_dict.items()
            }
        else:
            self.mass_samples_dict = dict()
            self.mass_min_max_dict = dict()

    def _asset_ids_from_contains(self, contains_list):
        '''
        Extracts GSO asset IDs from a list of patterns to match that may be part of the asset name.
        '''
        ids_list = self.gso_source.all_asset_ids
        ids_list = [
            asset_id for asset_id in ids_list
            if any([pattern in asset_id.lower() for pattern in contains_list])
        ]
        return ids_list

    def _get_random_asset_of_kind(self, object_kind, allow_complex=True):
        upside_down = False

        if object_kind == 'any':
            # Manage dataset splits as a function of phase, if applicable.
            if self.split_objects:
                (train_ids,
                 test_ids) = self.gso_source.get_test_split(fraction=0.1)
                if self.phase == 'train':
                    active_subset = train_ids
                elif self.phase == 'test':
                    active_subset = test_ids
            else:
                active_subset = self.gso_source.all_asset_ids

        elif object_kind == 'container':
            # In the general container case, force 5% to be hats, 10% to be shoes, and the rest any.
            dice = self.random_state.rand()
            if allow_complex and dice < 0.05:
                active_subset = GSO_HAT_IDS
                upside_down = True
            elif allow_complex and dice < 0.15:
                active_subset = self.gso_shoe_ids
            else:
                active_subset = GSO_CONTAINER_IDS

        elif object_kind == 'carrier':
            active_subset = GSO_CARRIER_IDS

        elif object_kind == 'box':
            active_subset = self.gso_box_ids

        else:
            raise ValueError(object_kind)

        asset_id = self.random_state.choice(active_subset)
        return (asset_id, upside_down)

    def prepare_next_scene(self,
                           phase,
                           random_seed,
                           camera_radius_range=(11.0, 16.0),
                           focal_length=35.0):
        '''
        :param phase (str): train / val / test.
        :param random_seed (int): Random seed to use for this scene.
        :param camera_radius_range (tuple of float): (min, max) radius from center of scene of
            random camera trajectory.
        :param focal_length (float): Focal length of camera in mm.
        '''
        # NOTE: Both val_aug and val_noaug will use the train background / object splits!
        # Test splits are therefore reserved for the actual test set only.
        if 'val' in phase:
            phase = 'train'
        assert phase in ['train', 'test']

        start_time = time.time()
        self.phase = phase
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(seed=random_seed)

        # Instantiate actual Kubric scene object.
        self.scene = self.kb_module.Scene(frame_start=0,
                                          frame_end=self.num_frames - 1,
                                          frame_rate=self.frame_rate,
                                          resolution=(self.frame_width,
                                                      self.frame_height))

        # =============================
        # Prepare PyBullet and Blender.

        # Reuse simulator instance throughout the entire process lifetime to avoid PyBullet errors.
        if self.simulator is None:
            self.simulator = self.kb_module.simulator.PyBullet(
                self.scene, scratch_dir=self.scratch_dir)
        else:
            self.simulator.scene = self.scene
            self.simulator.scratch_dir = self.scratch_dir

        # Refresh renderer every time to ensure we start fresh.
        motion_blur_value = self.random_state.uniform(
            0.4, 0.8) if self.motion_blur else 0.0
        self.renderer = self.kb_module.renderer.Blender(
            self.scene,
            scratch_dir=self.scratch_dir,
            adaptive_sampling=False,
            use_denoising=True,
            samples_per_pixel=self.render_samples_per_pixel,
            motion_blur=motion_blur_value)

        # Manage dataset splits as a function of phase, if applicable.
        if self.split_backgrounds:
            (train_ids,
             test_ids) = self.hdri_source.get_test_split(fraction=0.1)
            if self.phase == 'train':
                active_subset = train_ids
            elif self.phase == 'test':
                active_subset = test_ids
        else:
            active_subset = self.hdri_source.all_asset_ids

        # ====================================
        # Populate scene with HDRI background.
        hdri_id = self.random_state.choice(active_subset)
        background_hdri = self.hdri_source.create(asset_id=hdri_id,
                                                  name='bg_hdri')
        self.scene.metadata['background'] = hdri_id

        # Paint dome object (half sphere) with the selected image.
        self.dome = self.kubasic_source.create(asset_id='dome',
                                               name='dome',
                                               friction=1.0,
                                               restitution=0.0,
                                               static=True,
                                               background=True)
        assert isinstance(self.dome, self.kb_module.FileBasedObject)
        self.scene += self.dome

        # Apply random yaw because camera position may be fixed.
        # NOTE: Disabled for kubcon_v9 because it causes misalignment with HDRI ambient light.
        # self.kb_module.rotation_sampler(axis='Z')(self.dome, self.random_state)

        dome_blender = self.dome.linked_objects[self.renderer]
        texture_node = dome_blender.data.materials[0].node_tree.nodes[
            'Image Texture']
        texture_node.image = self.bpy_module.data.images.load(
            background_hdri.filename)

        # Subdivide the dome to increase vertex density
        self.bpy_module.context.view_layer.objects.active = dome_blender
        self.bpy_module.ops.object.mode_set(mode='EDIT')
        self.bpy_module.ops.mesh.select_all(action='SELECT')
        self.bpy_module.ops.mesh.subdivide(
            number_cuts=2)  # Increase this for more vertices
        self.bpy_module.ops.object.mode_set(mode='OBJECT')
        # Make the illumination and shadows consistent with the background.
        self.renderer._set_ambient_light_hdri(background_hdri.filename)

        # ========================
        # Setup camera trajectory.
        self._setup_camera(camera_radius_range=camera_radius_range,
                           focal_length=focal_length,
                           first_time=True)

        return time.time() - start_time

    def _setup_camera(self,
                      start_yaw_deg=0.0,
                      camera_radius_range=(11.0, 16.0),
                      focal_length=35.0,
                      first_time=False):
        '''
        :param start_yaw_deg (float): Initial yaw angle in degrees.
        :param camera_radius_range (tuple of float): (min, max) radius from center of scene of
            random camera trajectory.
        '''
        # NOTE: 35, 32 corresponds to horizontal FoV of 49.1 degrees which matches Zero123 exactly.
        # OTOH, 32, 32 corresponds to a slightly higher horizontal FoV of 53.1 degrees.
        self.scene.camera = self.kb_module.PerspectiveCamera(
            focal_length=focal_length, sensor_width=32.0)

        if first_time:
            if self.max_camera_speed > 0.0:
                movement_speed = self.random_state.uniform(
                    0.0, self.max_camera_speed)
            else:
                movement_speed = 0.0

            # NOTE: Compared to random half sphere sampling (as is default in MOVi),
            # this alternative is slightly biased toward lower viewing angles.
            radius_min = camera_radius_range[0]
            radius_max = camera_radius_range[1]
            cam_start_radius = self.random_state.uniform(
                radius_min, radius_max)

            z_min = 0.2
            z_max = cam_start_radius - 0.8
            cam_start_z = self.random_state.uniform(z_min, z_max)

            xy_radius = np.sqrt((cam_start_radius**2 - cam_start_z**2))
            cam_start_x = xy_radius * np.cos(start_yaw_deg * np.pi / 180.0)
            cam_start_y = xy_radius * np.sin(start_yaw_deg * np.pi / 180.0)
            fix_start = (cam_start_x, cam_start_y, cam_start_z)

            # NOTE: z_offset here means the minimum z at every frame.
            (camera_start, camera_end) = self.sample_linear_camera_motion(
                movement_speed,
                inner_radius=radius_min,
                outer_radius=radius_max,
                z_offset=z_min / 2.0,
                fix_start=fix_start)

            # NOTE: In Kubric-4D / multi-view, we always look at
            # precisely 1m above the ground at the center of the scene.
            xyz_look = np.array([0.0, 0.0, 1.0])

        else:
            # We wish to replicate all movement precisely, just from a different azimuth angle.
            rel_yaw_deg = start_yaw_deg - self.last_start_yaw_deg
            camera_start = _rotate_yaw(self.last_camera_start, rel_yaw_deg)
            camera_end = _rotate_yaw(self.last_camera_end, rel_yaw_deg)
            xyz_look = _rotate_yaw(self.last_xyz_look, rel_yaw_deg)

        # Linearly interpolate the camera position between these two points while keeping it focused
        # on the center of the scene. We start one frame early and end one frame late to ensure that
        # forward and backward optical flow remain consistent for the last and first frames.
        for frame in range(-1, self.num_frames + 2):
            interp = ((frame + 1) / (self.num_frames + 3))
            self.scene.camera.position = (
                (1.0 - interp) * np.array(camera_start) +
                interp * np.array(camera_end))
            self.scene.camera.look_at(xyz_look)
            self.scene.camera.keyframe_insert('position', frame)
            self.scene.camera.keyframe_insert('quaternion', frame)

        # Save parameters for possible rotations in the future.
        self.last_start_yaw_deg = start_yaw_deg
        self.last_camera_start = camera_start
        self.last_camera_end = camera_end
        self.last_xyz_look = xyz_look

    def set_camera_yaw(self, new_yaw_deg):
        '''
        :param new_yaw_deg (float): New yaw angle in degrees.
        '''
        start_time = time.time()

        self._setup_camera(start_yaw_deg=new_yaw_deg, first_time=False)

        return time.time() - start_time

    def write_camera_intrinsics(self,
                                output_dir: str,
                                filename: str = "camera_intrinsics.json"):
        # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        W, H = self.frame_width, self.frame_height
        # 1) get physical focal length & sensor width
        f = self.scene.camera.focal_length  # in mm
        sw = self.scene.camera.sensor_width  # in mm
        # 2) compute derived sensor height to respect your W×H
        sh = sw * (H / W)

        # 3) normalized intrinsics in [–1,1] unit image coords
        fx = f / sw
        fy = f / sh
        px = 0.5
        py = 0.5

        K = [[fx, 0.0, -px], [0.0, -fy, -py], [0.0, 0.0, -1.0]]

        # 6) Package everything up
        intrinsics = {"K": K}

        # 7) Write to JSON
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w") as f:
            json.dump(intrinsics, f, indent=2)
        print(f"Camera intrinsics written to {out_path}")

    def setup_camera_exact(self,
                           start_position,
                           start_look_at,
                           end_position=None,
                           end_look_at=None,
                           focal_length=35.0):
        '''
        This much simpler method is necessary for tightly controlled camera trajectories,
            for example fixed alternating poses per viewpoint.
        :param start_position (tuple of float): (x, y, z) coordinates of camera 3D starting position.
        :param start_look_at (tuple of float): (x, y, z) coordinates of camera 3D initial look-at.
        :param end_position (tuple of float): (x, y, z) coordinates of camera 3D ending position.
        :param end_look_at (tuple of float): (x, y, z) coordinates of camera 3D final look-at.
        :param focal_length (float): Focal length of camera in mm.
        '''
        start_time = time.time()

        start_position = np.array(start_position)
        start_look_at = np.array(start_look_at)
        if end_position is None:
            end_position = start_position
        else:
            end_position = np.array(end_position)
        if end_look_at is None:
            end_look_at = start_look_at
        else:
            end_look_at = np.array(end_look_at)

        self.scene.camera = self.kb_module.PerspectiveCamera(
            focal_length=focal_length, sensor_width=32.0)

        # Linearly interpolate the camera position between these two points while keeping it focused
        # on the center of the scene. We start one frame early and end one frame late to ensure that
        # forward and backward optical flow remain consistent for the last and first frames.
        for frame_idx in range(-1, self.num_frames + 2):
            time_val = frame_idx / self.num_frames  # Could be slightly < 0 or > 1!
            self.scene.camera.position = (
                1.0 - time_val) * start_position + time_val * end_position
            self.scene.camera.look_at((1.0 - time_val) * start_look_at +
                                      time_val * end_look_at)
            self.scene.camera.keyframe_insert('position', frame_idx)
            self.scene.camera.keyframe_insert('quaternion', frame_idx)

        return time.time() - start_time

    def _fix_mass_knowledge(self, asset_id, obj):
        auto_mass = obj.mass
        obj.metadata['auto_mass'] = obj.mass
        obj.metadata['auto_density'] = obj.mass / obj.metadata['volume']

        if asset_id in self.mass_min_max_dict:
            # Assume there is some inherent inaccuracy / uncertainty in the mass estimation, by
            # applying uniformly random noise within +/- 50% bounds.
            gpt_mass = max(
                self.random_state.uniform(*self.mass_min_max_dict[asset_id]),
                1e-4)

            # Apply geometric mean between the default and knowledge-derived values.
            obj.mass = np.sqrt(gpt_mass * auto_mass)

            obj.metadata['override_mass'] = obj.mass
            obj.metadata[
                'override_density'] = obj.mass / obj.metadata['volume']
            # print(f'Set mass to {obj.mass * 1000.0:.1f} g for {asset_id}, '
            #                  f'which amounts to a factor x{obj.mass / auto_mass:.1f} change in density.')

        elif len(self.mass_min_max_dict) != 0:
            # Always call this to update random_state to maintain consistency / reproducibility.
            stub_call = self.random_state.uniform(0.5, 1.5)
            print(f'[yellow]No mass knowledge for {asset_id}.')

        return obj

    def insert_static_objects(self,
                              min_count=8,
                              max_count=12,
                              force_containers=0,
                              force_carriers=0,
                              any_diameter_range=(0.75, 2.5),
                              container_carrier_diameter_range=(1.25, 3.0),
                              simple_containers_only=False):
        start_time = time.time()

        total_static_objects = self.random_state.randint(
            min_count, max_count + 1)
        num_any_objects = max(
            total_static_objects - force_containers - force_carriers, 0)
        object_kinds = ['container'] * force_containers + \
            ['carrier'] * force_carriers + \
            ['any'] * num_any_objects
        assert len(object_kinds) >= total_static_objects
        print(
            'Randomly placing %d containers, %d carriers, and %d static/any objects:',
            force_containers, force_carriers, num_any_objects)

        for i, object_kind in enumerate(object_kinds):
            # NOTE: While carrier is always carrier, container may be generic, hat, or shoe.
            # NOTE: Dataset split handling only happens if object_kind is any.
            (asset_id, upside_down) = self._get_random_asset_of_kind(
                object_kind, allow_complex=not (simple_containers_only))
            obj = self.gso_source.create(asset_id=asset_id,
                                         name=f'static_{i:03d}')
            assert isinstance(obj, self.kb_module.FileBasedObject)

            # Overwrite mass with provided knowledge, if available.
            obj = self._fix_mass_knowledge(asset_id, obj)

            # Make all objects of roughly similar size, though containers and carriers should be
            # slightly bigger.
            axis_diameter = self.random_state.uniform(*any_diameter_range) \
                if object_kind == 'any' else \
                self.random_state.uniform(*container_carrier_diameter_range)
            scale_factor = axis_diameter / np.max(obj.bounds[1] -
                                                  obj.bounds[0])
            obj.scale = scale_factor
            obj.metadata['axis_diameter'] = axis_diameter
            obj.metadata['scale_factor'] = scale_factor
            obj.metadata['mass_pre'] = obj.mass
            # Volumetric scaling law is important here.
            obj.mass *= np.power(scale_factor, self.mass_scaling_law)
            obj.metadata['mass_post'] = obj.mass

            # Turn hats upside down. The initial quaternion is (w, x, y, z) = (1, 0, 0, 0).
            if upside_down and object_kind in ['container', 'carrier']:
                # This rotates 180 degrees around X, so flips Y and Z.
                obj.quaternion = np.array([0, 1, 0, 0])
            obj.metadata['initial_quaternion'] = obj.quaternion

            # Insert object into scene, ensuring that all contents remain disjoint.
            # NOTE: For containers and carriers to work, they must stay mostly upright when placed.
            self.scene += obj
            rotation_axis = 'Z' if object_kind in ['container', 'carrier'
                                                   ] else None
            spawn_region = STATIC_SPAWN_REGION if object_kind == 'any' else \
                CONTAINER_CARRIER_SPAWN_REGION if object_kind in ['container', 'carrier'] else None
            self.move_until_no_overlap(obj,
                                       self.simulator,
                                       rotation_axis=rotation_axis,
                                       spawn_region=spawn_region,
                                       rng=self.random_state)

            obj.friction = 1.0
            obj.restitution = 0.0
            obj.metadata['insert_order'] = i
            obj.metadata['object_kind'] = object_kind
            obj.metadata['is_dynamic'] = False

        return time.time() - start_time

    def insert_dynamic_objects(self,
                               min_count=4,
                               max_count=6,
                               force_boxes=0,
                               any_diameter_range=(0.5, 2.0),
                               box_diameter_range=(0.75, 2.0)):
        start_time = time.time()

        total_dynamic_objects = self.random_state.randint(
            min_count, max_count + 1)
        num_any_objects = max(total_dynamic_objects - force_boxes, 0)
        object_kinds = ['box'] * force_boxes + \
            ['any'] * num_any_objects
        assert len(object_kinds) >= total_dynamic_objects
        print('Randomly placing %d boxes and %d dynamic/any objects:',
              force_boxes, num_any_objects)

        for i, object_kind in enumerate(object_kinds):
            # NOTE: While carrier is always carrier, container may be generic, hat, or shoe.
            # NOTE: Dataset split handling only happens if object_kind is any.
            (asset_id,
             upside_down) = self._get_random_asset_of_kind(object_kind)
            obj = self.gso_source.create(asset_id=asset_id,
                                         name=f'dynamic_{i:03d}')
            assert isinstance(obj, self.kb_module.FileBasedObject)

            # Overwrite mass with provided knowledge, if available.
            obj = self._fix_mass_knowledge(asset_id, obj)

            # Make all objects of roughly similar size, though dynamic objects should be slightly
            # smaller than static ones, especially containers.
            axis_diameter = self.random_state.uniform(*any_diameter_range) \
                if object_kind == 'any' else \
                self.random_state.uniform(*box_diameter_range)
            scale_factor = axis_diameter / np.max(obj.bounds[1] -
                                                  obj.bounds[0])
            obj.scale = scale_factor
            obj.metadata['axis_diameter'] = axis_diameter
            obj.metadata['scale_factor'] = scale_factor
            obj.metadata['mass_pre'] = obj.mass
            # Volumetric scaling law is important here.
            obj.mass *= np.power(scale_factor, self.mass_scaling_law)
            obj.metadata['mass_post'] = obj.mass

            # Insert object into scene, ensuring that all contents remain disjoint.
            self.scene += obj
            rotation_axis = 'Z' if object_kind in ['box'] else None
            spawn_region = DYNAMIC_SPAWN_REGION
            self.move_until_no_overlap(obj,
                                       self.simulator,
                                       rotation_axis=rotation_axis,
                                       spawn_region=spawn_region,
                                       rng=self.random_state)

            # Assign random horizontal velocity with a strong bias toward the center of the scene.
            init_vel_independent = self.random_state.uniform(*VELOCITY_RANGE)
            init_pull_direction = np.array(
                [obj.position[0], obj.position[1], 0.0])
            init_pull_factor = self.random_state.uniform(0.6, 1.2)
            init_vel_combined = init_vel_independent - init_pull_direction * init_pull_factor
            obj.velocity = init_vel_combined

            # NOTE: This (friction & restitution) is new relative to kubcon_v7.
            obj.friction = self.random_state.uniform(
                *self.object_friction_range)
            obj.restitution = self.random_state.uniform(
                *self.object_restit_range)
            obj.metadata['init_pull_factor'] = init_pull_factor
            obj.metadata['insert_order'] = i
            obj.metadata['object_kind'] = object_kind
            obj.metadata['is_dynamic'] = True
            obj.metadata['is_snitch'] = False

        return time.time() - start_time

    def insert_snitch(self,
                      at_x=0.0,
                      at_y=0.0,
                      at_z=4.5,
                      vel_x=0.0,
                      vel_y=0.0,
                      vel_z=-3.5,
                      size_meters=0.35,
                      gso_asset_id='Vtech_Roll_Learn_Turtle'):
        start_time = time.time()

        obj = self.gso_source.create(asset_id=gso_asset_id, name='snitch')
        assert isinstance(obj, self.kb_module.FileBasedObject)

        # Assign fixed size to snitch; smaller than almost all other objects.
        axis_diameter = size_meters
        scale_factor = axis_diameter / np.max(obj.bounds[1] - obj.bounds[0])
        obj.scale = scale_factor
        obj.metadata['axis_diameter'] = axis_diameter
        obj.metadata['scale_factor'] = scale_factor

        # Insert snitch into scene at desired location.
        # NOTE: This happens without regard for intersections.
        self.scene += obj
        obj.position = np.array([at_x, at_y, at_z])
        obj.velocity = np.array([vel_x, vel_y, vel_z])
        obj.metadata['is_dynamic'] = True
        obj.metadata['is_snitch'] = True

        self.scene.metadata['insert_snitch_args'] = dict()
        self.scene.metadata['insert_snitch_args']['at'] = (at_x, at_y, at_z)
        self.scene.metadata['insert_snitch_args']['vel'] = (vel_x, vel_y,
                                                            vel_z)
        self.scene.metadata['insert_snitch_args']['size_meters'] = size_meters
        self.scene.metadata['insert_snitch_args'][
            'gso_asset_id'] = gso_asset_id

        return time.time() - start_time

    def reset_objects_velocity_friction_restitution(self):
        start_time = time.time()

        for obj in self.scene.foreground_assets:
            if hasattr(obj, 'velocity'):
                obj.velocity = np.array([0.0, 0.0, 0.0])
                obj.friction = self.random_state.uniform(
                    *self.object_friction_range)
                obj.restitution = self.random_state.uniform(
                    *self.object_restit_range)

        # Correct floor phyisical properties.
        # Set physics properties only for the ground (not the skybox)
        self.dome.friction = self.random_state.uniform(
            *self.dome_friction_range)
        self.dome.restitution = self.random_state.uniform(
            *self.dome_restit_range)

        return time.time() - start_time

    def perturb_object_positions(self, max_offset_meters=0.01):
        '''
        Randomly translate all foreground objects by uniformly random noise vectors.
        '''
        start_time = time.time()

        for obj in self.scene.foreground_assets:

            # Sample perturbation, but ensure we can't glitch into the floor.
            translation_x = self.random_state.uniform(-max_offset_meters,
                                                      max_offset_meters)
            translation_y = self.random_state.uniform(-max_offset_meters,
                                                      max_offset_meters)
            translation_z = self.random_state.uniform(0.0, max_offset_meters)
            translation_meters = np.array(
                [translation_x, translation_y, translation_z])

            # Apply offset to instance position.
            obj.position = obj.position + translation_meters
            obj.metadata['perturbation'] = translation_meters

        return time.time() - start_time

    def simulate_frames(self, frame_start, frame_end):
        '''
        :param frame_start (int): First frame (inclusive) to simulate.
        :param frame_end (int): Last frame (inclusive) to simulate.
        :return (animations, collisions, runtime).
            animations: ??.
            collisions: ??.
            runtime (float): Total runtime of this call in seconds.
        '''
        start_time = time.time()

        (animations, collisions) = self.simulator.run(frame_start, frame_end)

        self.last_animations = animations
        self.last_collisions = collisions

        self.last_data_stack = None  # Mark as invalidated (must call render_frames again).

        return (animations, collisions, time.time() - start_time)

    def render_frames(self,
                      frame_start,
                      frame_end,
                      return_layers=[
                          'rgba', 'forward_flow', 'depth', 'normal',
                          'object_coordinates', 'segmentation'
                      ]):
        '''
        :param frame_start (int): First frame (inclusive) to render.
        :param frame_end (int): Last frame (inclusive) to render.
        :param return_layers (list of str): Keys of data modalities to process.
        :return (data_stack, runtime).
            data_stack: Dictionary containing a subset or all of these items:
                rgba: (T, H, W, 4) uint8.
                forward_flow: (T, H, W, 2) float32.
                depth: (T, H, W, 1) float32.
                normal: (T, H, W, 3) uint16.
                object_coordinates: (T, H, W, 3) uint16.
                segmentation: (T, H, W, 1) uint32.
            runtime (float): Total runtime of this call in seconds.
        '''
        start_time = time.time()

        # Set number of CPU cores to use (again), if specified.
        # https://docs.blender.org/api/current/bpy.types.RenderSettings.html
        if self.render_cpu_threads >= 1:
            self.bpy_module.context.scene.render.threads_mode = 'FIXED'
            self.bpy_module.context.scene.render.threads = self.render_cpu_threads

        # Render the selected subset of frames.
        data_stack = self.renderer.render(frames=list(
            range(frame_start, frame_end + 1)),
                                          return_layers=return_layers)

        # Perform postprocessing to obtain higher-level annotations.
        # Rank such that id = 1 is visually the biggest / most visible.
        # NOTE: Somewhat counterintuitively, some instances will *never* be visible, so their
        # visibility value will be exactly 0 for all frames.
        self.kb_module.compute_visibility(data_stack['segmentation'],
                                          self.scene.assets)

        # NOTE: self.scene.assets includes dome, camera, etc, but self.scene.foreground_assets
        # does not. The object order is now preserved and follows the order of creation.
        self.foreground_assets = self.scene.foreground_assets
        data_stack['segmentation'] = self.kb_module.adjust_segmentation_idxs(
            data_stack['segmentation'], self.scene.assets,
            self.foreground_assets)
        self.scene.metadata['num_instances'] = len(self.foreground_assets)
        self.scene.metadata['num_valo_instances'] = len([
            x for x in self.foreground_assets
            if np.array(x.metadata['visibility']).max() > 0
        ])

        # I decided to discard sorting entirely,
        # since it became too messy and inconsistent in multiview settings.
        # Now how to interpret the segmentation array contents?
        # When the value of data_stack['segmentation'] is 5, we point to self.sorted_fg_assets[4].
        # Therefore we also point to metadata['instances'][4]; this list is created in get_metadata.
        # Finally, when the segmentation value is 0, this means background (i.e. dome) because it is
        # not in the list of instances / foreground assets.

        # Store results for if we want to obtain metadata later.
        self.last_data_stack = data_stack

        self.last_div_data = None  # Mark as unused or not yet calculated.

        return (data_stack, time.time() - start_time)

    def render_frames_divided_objects(
            self,
            frame_start,
            frame_end,
            return_layers=['rgba', 'depth', 'segmentation']):
        '''
        :param frame_start (int): First frame (inclusive) to render.
        :param frame_end (int): Last frame (inclusive) to render.
        Isolates all foreground objects that (even those that are not visible at least once
            / VALO) and renders a full video separately for each of them (by making all others
            invisible), in order to generate complete instance segmentation masks that always
            persist through occlusions otherwise caused by other objects in the input video.
            NOTE: This functionality is by definition very thorough and therefore expensive.
        :return (div_data, runtime).
            div_data (dict): Similar to data_stack.
                rgba: (T, H, W, 4, K) uint8.
                depth: (T, H, W, 1, K) float32.
                segmentation: (T, H, W, 1, K) uint8.
                For every array, channel dimension follows the order of foreground_assets.
            runtime (float): Total runtime of this call in seconds.
        '''
        start_time = time.time()

        (T, H, W, _) = self.last_data_stack['segmentation'].shape
        # K = len(self.sorted_valo_fg_assets)
        K = len(self.foreground_assets)

        object_scales = copy.deepcopy(
            [asset.scale for asset in self.foreground_assets])
        raw_stacks = []
        print(f'Called render_frames_divided_objects with {K} instances...')

        for k in range(K):
            # Make all objects invisible except the current one.
            # NOTE: There is no direct visibility flag available, so we have to resize it into
            # nothing instead.
            for asset in self.foreground_assets:
                asset.scale = 0.0
            self.foreground_assets[k].scale = copy.deepcopy(object_scales[k])

            # Re-render the selected subset of frames.
            data_stack = self.renderer.render(frames=list(
                range(frame_start, frame_end + 1)),
                                              return_layers=return_layers)

            # Reorder IDs to match render_frames(), such that the array values should always become
            # k + 1 at iteration k.
            data_stack[
                'segmentation'] = self.kb_module.adjust_segmentation_idxs(
                    data_stack['segmentation'], self.scene.assets,
                    self.foreground_assets)

            # Store current render output.
            raw_stacks.append(data_stack)

        # Construct array with combined results.
        div_data = dict()
        for layer in return_layers:
            div_data[layer] = np.stack(
                [data_stack[layer] for data_stack in raw_stacks], axis=-1)

        self.last_div_data = div_data

        return (div_data, time.time() - start_time)

    def get_metadata(self, exclude_collisions=False):
        '''
        :param exclude_collisions (bool): Exclude collisions from metadata.
        '''
        start_time = time.time()

        self.kb_module.post_processing.compute_bboxes(
            self.last_data_stack['segmentation'], self.foreground_assets)

        metadata = dict()
        metadata['scene'] = self.kb_module.get_scene_metadata(self.scene)
        metadata['camera'] = self.kb_module.get_camera_info(self.scene.camera)
        metadata['instances'] = self.kb_module.get_instance_info(
            self.scene, self.foreground_assets)

        # NOTE: collisions generate a rather huge amount of information (~20 MB), so we omit them
        # for views beyond the first one.
        if not exclude_collisions:
            metadata['collisions'] = self.kb_module.process_collisions(
                self.last_collisions,
                self.scene,
                assets_subset=self.foreground_assets)

        metadata['dome'] = self.kb_module.get_instance_info(
            self.scene, [self.dome])[0]
        metadata['random_seed'] = self.random_seed

        # Store results for if we want to obtain annotations later.
        self.last_metadata = metadata

        return (metadata, time.time() - start_time)

    def write_all_data(self, output_dir):
        start_time = time.time()

        os.makedirs(output_dir, exist_ok=True)
        self.kb_module.write_image_dict(self.last_data_stack,
                                        output_dir)  # , is_divided=False)

        if self.last_div_data is not None:
            self.kb_module.write_image_dict(self.last_div_data,
                                            output_dir)  # , is_divided=True)

        return time.time() - start_time

    def write_simulator_state(self, dst_fp):
        start_time = time.time()

        assert dst_fp.lower().endswith('.bullet')
        self.simulator.save_state(dst_fp)

        return time.time() - start_time

    def write_renderer_state(self, dst_fp):
        start_time = time.time()

        assert dst_fp.lower().endswith('.blend')
        self.renderer.save_state(dst_fp)

        return time.time() - start_time

    def sample_linear_camera_motion(self,
                                    movement_speed: float,
                                    inner_radius: float = 9.0,
                                    outer_radius: float = 12.0,
                                    z_offset: float = 0.1,
                                    fix_start=None):
        '''
        Sample a linear path which starts and ends within a half-sphere shell.
        '''
        for _ in range(1024):
            if fix_start is None:
                camera_start = np.array(
                    self.kb_module.sample_point_in_half_sphere_shell(
                        inner_radius, outer_radius, z_offset))
            else:
                camera_start = np.array(fix_start)

            if movement_speed <= 0.0:
                camera_end = camera_start

            else:
                direction = self.random_state.rand(3) - 0.5
                movement = direction / (np.linalg.norm(direction) +
                                        1e-7) * movement_speed
                camera_end = camera_start + movement

            if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius
                    and camera_end[2] >= z_offset):
                return (camera_start, camera_end)

        raise RuntimeError('Could not find a valid camera path.')

    def move_until_no_overlap(self,
                              asset,
                              simulator,
                              rotation_axis=None,
                              spawn_region=((-1, -1, -1), (1, 1, 1)),
                              max_trials=100,
                              rng=None):
        return self.kb_module.resample_while(
            asset,
            samplers=[
                self.kb_module.rotation_sampler(axis=rotation_axis),
                self.kb_module.position_sampler(spawn_region)
            ],
            condition=simulator.check_overlap,
            max_trials=max_trials,
            rng=rng)

    def quaternion_matrix(self, q_wxyz):
        """
        Convert a Kubric quaternion [w, x, y, z] into 
        a 4×4 homogeneous transform matrix.
        """
        # reorder into [x, y, z, w] for SciPy:
        q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
        R3 = R.from_quat(q_xyzw).as_matrix()  # shape (3,3)
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R3
        return M

    def bake_vertex_colors(self, bpy, obj, is_dome=False):
        sc = bpy.context.scene

        # —————————————————————————————
        # 1) Configure Cycles & the bake itself
        # —————————————————————————————
        sc.render.engine = 'CYCLES'
        sc.cycles.samples = 64
        sc.cycles.bake_type = 'DIFFUSE'
        sc.render.bake.target = 'VERTEX_COLORS'
        if is_dome:  # HDRI dome → bake raw pixels
            sc.cycles.bake_type = 'DIFFUSE'
            sc.render.bake.use_pass_color = True
            sc.render.bake.use_pass_direct = True
            sc.render.bake.use_pass_indirect = True
        else:  # foreground mesh → bake lighting
            sc.cycles.bake_type = 'DIFFUSE'
            sc.render.bake.use_pass_color = True
            sc.render.bake.use_pass_direct = True
            sc.render.bake.use_pass_indirect = True

        # —————————————————————————————
        # 2) Ensure & activate your LitColor v-color layer
        # —————————————————————————————
        mesh = obj.data
        layer_name = 'LitColor'
        if layer_name not in mesh.vertex_colors:
            mesh.vertex_colors.new(name=layer_name)
        idx = mesh.vertex_colors.find(layer_name)
        mesh.vertex_colors.active_index = idx

        # —————————————————————————————
        # 3) Make sure we’re in Object mode with this object active
        # —————————————————————————————
        # deselect everything
        bpy.ops.object.select_all(action='DESELECT')
        # select and activate
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        # switch to Object mode (the bake operator needs it)
        bpy.ops.object.mode_set(mode='OBJECT')

        # —————————————————————————————
        # 4) Bake into vertex-colors
        # —————————————————————————————
        bpy.ops.object.bake(type='DIFFUSE', target='VERTEX_COLORS')

        # —————————————————————————————
        # 5) Clean up selection
        # —————————————————————————————
        obj.select_set(False)

    def save_hdri_file(self, out):
        orig_path = Path(self.hdri_filename)
        # define where you want the copy to live
        dest_dir = Path(out)
        dest_dir.mkdir(parents=True, exist_ok=True)
        # keep same filename (or change it here)
        dest_path = Path(out) / f"frame.hdr"
        # perform the actual file copy (including metadata)
        shutil.copy2(orig_path, dest_path)

    def export_mesh_sequence(self, out_dir):
        import bpy
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        all_assets = list(self.scene.foreground_assets +
                          self.scene.background_assets)
        for f in range(self.scene.frame_start, self.scene.frame_end + 1):
            bpy.context.scene.frame_set(f)

            # 1) bake each object's vertex colors
            for obj in all_assets:
                if not isinstance(obj, self.kb_module.FileBasedObject):
                    continue
                bo = obj.linked_objects[self.renderer]
                if obj is self.dome:
                    self.bake_vertex_colors(bpy, bo, True)
                else:
                    self.bake_vertex_colors(bpy, bo, False)

            # — remove dome faces above Z=12 (instead of deleting vertices) —
            dome_obj = self.dome.linked_objects[self.renderer]
            bpy.context.view_layer.objects.active = dome_obj

            # 1) Go into Edit mode and set face-select, then clear selection
            self.bpy_module.ops.object.mode_set(mode='EDIT')
            self.bpy_module.ops.mesh.select_mode(type='FACE')
            self.bpy_module.ops.mesh.select_all(action='DESELECT')

            # 2) Back to Object mode to mark faces above Z=12
            self.bpy_module.ops.object.mode_set(mode='OBJECT')
            for poly in dome_obj.data.polygons:
                poly.select = (poly.center.z > 1.0)

            # 3) Edit mode again to delete those faces
            self.bpy_module.ops.object.mode_set(mode='EDIT')
            self.bpy_module.ops.mesh.delete(type='FACE')

            # 4) Return to Object mode
            self.bpy_module.ops.object.mode_set(mode='OBJECT')

            # — now select & export as before —
            bpy.ops.object.select_all(action='DESELECT')
            for obj in all_assets:
                bo = obj.linked_objects[self.renderer]
                bo.select_set(True)
            active = all_assets[0]
            bpy.context.view_layer.objects.active = active.linked_objects[
                self.renderer]

            # 3) export them in one PLY (Blender will merge your selection under the hood)
            filepath = out / f"frame_{f:04d}.ply"
            bpy.ops.export_mesh.ply(filepath=str(filepath),
                                    use_selection=True,
                                    use_normals=True,
                                    use_colors=True,
                                    use_ascii=False)

            # 2) Re-compress with LZMA (XZ) at maximum+extreme preset
            xz_path = filepath.with_suffix(".ply.xz")
            # preset: 0–9 (higher = slower/more) | lzma.PRESET_EXTREME for extra‐slow maximum
            preset = 9 | lzma.PRESET_EXTREME
            with open(str(filepath),
                      "rb") as f_in, lzma.open(str(xz_path),
                                               "wb",
                                               preset=preset) as f_out:
                shutil.copyfileobj(f_in, f_out)

            filepath.unlink()


def _rotate_yaw(xyz, yaw_deg):
    xyz = xyz.astype(np.float64)
    yaw_rad = yaw_deg * np.pi / 180.0
    rotation_matrix = np.array(
        [[np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
         [np.sin(yaw_rad), np.cos(yaw_rad), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64)
    xyz_new = np.matmul(xyz, rotation_matrix.T)
    xyz_new = xyz_new.astype(np.float32)
    return xyz_new
