'''
Created by Basile Van Hoorick for GCD, 2024.
Generates multi-view Kubric videos.
'''

import os  # noqa
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

# Internal imports.
import data_utils

np.set_printoptions(precision=3, suppress=True)


def main(root_dp='/path/to/kubric_mv_dbg',
         mass_est_fp='gpt_mass_v4.txt',
         num_scenes=100,
         start_idx=0,
         end_idx=99999,
         num_workers=16,
         restart_count=30,
         seed=400,
         num_perturbs=1,
         num_views=16,
         frame_width=384,
         frame_height=256,
         num_frames=30,
         frame_rate=12,
         motion_blur=0,
         save_depth=1,
         save_coords=1,
         save_bkfw=0,
         render_samples_per_pixel=16,
         render_use_gpu=0,
         max_camera_speed=8.0,
         focal_length=32.0,
         fixed_alter_poses=1,
         few_views=4):

    # ==================================
    # CUSTOMIZE DATASET PARAMETERS HERE:

    start_idx = int(start_idx)
    end_idx = int(end_idx)

    perturbs_first_scenes = 0  # Only test.
    views_first_scenes = 999999  # Only test.
    test_first_scenes = 0  # For handling background & object asset splits (optional).

    root_dn = os.path.basename(root_dp)
    ignore_if_exist = True  # Already generated scene folders will be skipped.

    min_static = 6  # Kubric / MOVi = 10.
    max_static = 16  # Kubric / MOVi = 20.
    min_dynamic = 1  # Kubric / MOVi = 1.
    max_dynamic = 6  # Kubric / MOVi = 3.

    camera_radius_range = [12.0, 16.0]  # In meters.
    static_diameter_range = [1.0, 2.75]  # Kubric / MOVi = [0.75, 3.0].
    dynamic_diameter_range = [1.0, 2.75]  # Kubric / MOVi = [0.75, 3.0].

    fixed_radius = 15.0  # In meters.
    fixed_elevation_many = 5  # Primary target distribution.
    fixed_elevation_few = 45  # Look inside containers / behind occluders.
    fixed_look_at = [0.0, 0.0, 1.0]  # Matches _setup_camera().

    # NOTE: Because of /tmp size problems, we must stop and restart this script to empty the /tmp
    # directory inbetween runs. This counter indicates when all threads should finish.
    total_scn_cnt = mp.Value('i', 0)

    os.makedirs(root_dp, exist_ok=True)

    def do_scene(worker_idx, scene_idx, scene_dp, scene_dn):

        # Assign resources (which CPU and/or GPU I can use).
        data_utils.update_os_cpu_affinity(scene_idx % 4, 4)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(worker_idx % 8)

        # WARNING / NOTE: We CANNOT import bpy outside of the actual thread / process using it!
        import kubric as kb
        import kubric_sim
        import pybullet as pb

        render_cpu_threads = 4  #int(np.ceil(mp.cpu_count() / max(num_workers, 4)))
        print(
            f'{worker_idx}: {scene_idx}: Using {render_cpu_threads} CPU threads for rendering.'
        )

        np.random.seed(seed + scene_idx * 257)
        scratch_dir = f'/dss/dsstbyfs02/scratch/00/di97nip/tmp/mygenkub_{root_dn}/{scene_idx:05d}_{np.random.randint(10000, 99999)}'

        # NOTE: This instance must only be created once per process!
        my_kubric = kubric_sim.MyKubricSimulatorRenderer(
            frame_width=frame_width,
            frame_height=frame_height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            motion_blur=motion_blur,
            render_samples_per_pixel=render_samples_per_pixel,
            render_use_gpu=render_use_gpu,
            render_cpu_threads=render_cpu_threads,
            scratch_dir=scratch_dir,
            mass_est_fp=mass_est_fp,
            max_camera_speed=max_camera_speed,
            mass_scaling_law=2.0)

        os.makedirs(scene_dp, exist_ok=True)

        start_time = time.time()

        phase = 'test' if scene_idx < test_first_scenes else 'train'
        t = my_kubric.prepare_next_scene(
            phase,
            seed + scene_idx,
            camera_radius_range=camera_radius_range,
            focal_length=focal_length)
        print(f'{worker_idx}: {scene_idx}: prepare_next_scene took {t:.2f}s')

        t = my_kubric.insert_static_objects(
            min_count=min_static,
            max_count=max_static,
            any_diameter_range=static_diameter_range)
        print(
            f'{worker_idx}: {scene_idx}: insert_static_objects took {t:.2f}s')

        beforehand = int(round(4 * frame_rate))
        (_, _, t) = my_kubric.simulate_frames(-beforehand, -1)
        print(f'{worker_idx}: {scene_idx}: simulate_frames took {t:.2f}s')

        t = my_kubric.reset_objects_velocity_friction_restitution()
        print(f'{worker_idx}: {scene_idx}: '
              f'reset_objects_velocity_friction_restitution took {t:.2f}s')

        t = my_kubric.insert_dynamic_objects(
            min_count=min_dynamic,
            max_count=max_dynamic,
            any_diameter_range=dynamic_diameter_range)
        print(
            f'{worker_idx}: {scene_idx}: insert_dynamic_objects took {t:.2f}s')

        all_data_stacks = []
        all_videos = []

        # Determine multiplicity of this scene based on index.
        used_num_perturbs = num_perturbs if scene_idx < perturbs_first_scenes else 1
        used_num_views = num_views if scene_idx < views_first_scenes else 1
        start_yaw = my_kubric.random_state.uniform(0.0, 360.0)

        # Loop over butterfly effect variations.
        for perturb_idx in range(used_num_perturbs):

            print()
            print(
                f'{worker_idx}: {scene_idx}: '
                f'perturb_idx: {perturb_idx} / used_num_perturbs: {used_num_perturbs}'
            )
            print()

            # Ensure that the simulator resets its state for every perturbation.
            if perturb_idx == 0 and used_num_perturbs >= 2:
                print(f'Saving PyBullet simulator state...')
                # https://github.com/bulletphysics/bullet3/issues/2982
                pb.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
                pb_state = pb.saveState()

            elif perturb_idx >= 1:
                print(
                    f'{worker_idx}: {scene_idx}: Restoring PyBullet simulator state...'
                )
                pb.restoreState(pb_state)

            # Always simulate a little bit just before the actual starting point to ensure Kubric
            # updates its internal state (in particular, object positions) properly.
            (_, _, t) = my_kubric.simulate_frames(-1, 0)
            print(f'{worker_idx}: {scene_idx}: simulate_frames took {t:.2f}s')

            if used_num_perturbs >= 2:
                t = my_kubric.perturb_object_positions(max_offset_meters=0.005)
                print(
                    f'{worker_idx}: {scene_idx}: perturb_object_positions took {t:.2f}s'
                )

            (_, _, t) = my_kubric.simulate_frames(0, num_frames)
            print(f'{worker_idx}: {scene_idx}: simulate_frames took {t:.2f}s')
            #export mesh
            my_kubric.export_mesh_sequence(scene_dp)
            my_kubric.write_camera_intrinsics(scene_dp)

    def worker(worker_idx, num_workers, total_scn_cnt):

        machine_name = platform.node()
        log_name = f'{root_dn}_{machine_name}'

        my_start_idx = worker_idx + start_idx
        my_end_idx = min(num_scenes, end_idx)

        for scene_idx in range(my_start_idx, my_end_idx, num_workers):

            scene_dn = f'scn{scene_idx:05d}'
            scene_dp = os.path.join(root_dp, scene_dn)

            print()
            print(
                f'{worker_idx}: scene_idx: {scene_idx} / scene_dn: {scene_dn}')
            print()

            # Determine multiplicity of this scene based on index.
            used_num_perturbs = num_perturbs if scene_idx < perturbs_first_scenes else 1
            used_num_views = num_views if scene_idx < views_first_scenes else 1

            # Check for the latest file that could have been written.
            dst_json_fp = os.path.join(scene_dp, f'camera_intrinsics.json')
            if ignore_if_exist and os.path.exists(dst_json_fp):
                print(
                    f'{worker_idx}: This scene already exists at {dst_json_fp}, skipping!'
                )
                continue

            else:
                # We perform the actual generation in a separate thread to try to ensure that
                # no memory leaks survive.
                do_scene(worker_idx, scene_idx, scene_dp, scene_dn)
                #p = mp.Process(target=do_scene,
                #               args=(worker_idx, scene_idx, scene_dp,
                #                     scene_dn))
                #p.start()
                #p.join()

        print()
        print(f'I am done!')
        print()

        pass

    if num_workers <= 1:

        worker(0, 1, total_scn_cnt)

    else:

        processes = [
            mp.Process(target=worker,
                       args=(worker_idx, num_workers, total_scn_cnt))
            for worker_idx in range(num_workers)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    pass


if __name__ == '__main__':

    fire.Fire(main)

    pass
