import open3d as o3d
import numpy as np
import argparse

def _sample_random_view(elevation_range=(35.0, 35.0),
                        azimuth_range=(140.0, 140.0),
                        radius_range=(15.0, 15.0)):  #
    """
        Samples a random camera view defined by azimuth, elevation, and radius within specified ranges.

        Args:
            - None (uses self.azimuth_range, self.elevation_range, self.radius_range, and self.elevation_sample_sin).

        Returns:
            - azimuth (float): Randomly sampled azimuth angle (degrees) within self.azimuth_range.
            - elevation (float): Randomly sampled elevation angle (degrees) within self.elevation_range.
                            If self.elevation_sample_sin is True, samples uniformly in sine space for correct distribution.
            - radius (float): Randomly sampled radius within self.radius_range.
        """
    if azimuth_range[1] - azimuth_range[0] <= 0.0:
        azimuth = azimuth_range[0]
    else:
        azimuth = np.random.uniform(*azimuth_range)

    if elevation_range[1] - elevation_range[0] <= 0.0:
        elevation = elevation_range[0]
    else:
        elevation = np.random.uniform(*elevation_range)

    if radius_range[1] - radius_range[0] <= 0.0:
        radius = radius_range[0]
    else:
        radius = np.random.uniform(*radius_range)

    return azimuth, elevation, radius


def extrinsics_from_look_at(camera_position, camera_look_at):
    '''
    :param camera_position: (3) array of float.
    :param camera_look_at: (3) array of float.
    :return RT: (4, 4) array of float.
    '''
    # NOTE: In my convention (including Kubric and ParallelDomain),
    # the columns (= camera XYZ axes) should be: right, down, forward.

    # Calculate forward vector: Z.
    forward = (camera_look_at - camera_position)
    forward /= np.linalg.norm(forward)
    # Assume world's down vector: Y.
    world_down = np.array([0, 0, -1])
    # Calculate right vector: X = Y cross Z.
    right = np.cross(world_down, forward)
    right /= np.linalg.norm(right)
    # Calculate actual down vector: Y = Z cross X.
    down = np.cross(forward, right)

    # Construct 4x4 extrinsics matrix.
    RT = np.eye(4)
    RT[0:3, 0:3] = np.stack([right, down, forward], axis=1)
    RT[0:3, 3] = camera_position

    return RT


def cartesian_from_spherical(spherical, deg2rad=False):
    '''
    :param spherical: (..., 3) array of float.
    :return cartesian: (..., 3) array of float.
    '''
    azimuth = spherical[..., 0]
    elevation = spherical[..., 1]
    radius = spherical[..., 2]
    if deg2rad:
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    cartesian = np.stack([x, y, z], axis=-1)
    return cartesian


def calculate_se3fromAER(views):
    """
    Converts a batch of view parameters (azimuth, elevation, radius) into SE(3) camera extrinsic matrices.

    Args:
        - views (np.ndarray): Array of shape (Tcm, views_per_timestep, 3) where each entry is (azimuth, elevation, radius) in degrees.
                            Converted to Cartesian coordinates internally.
    
    Returns:
        - extrinsics (torch.Tensor): Tensor of shape (Tcm, views_per_timestep, 4, 4) containing camera-to-world matrices
                                    computed via a look-at transformation, with cameras looking at [0, 0, 1].
    """
    Tcm = 1
    views_cartesian = cartesian_from_spherical(views, deg2rad=True)
    views_cartesian[..., 2] += 1.0
    views_cartesian = np.expand_dims(views_cartesian, axis=0)

    look_at = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    look_at = np.tile(look_at[None], (Tcm, 1, 1))

    # Convert all to camera extrinsics over time.
    extrinsics = np.zeros((Tcm, 1, 4, 4), dtype=np.float32)
    for t in range(0, Tcm):
        for j in range(1):
            extrinsics[t, j] = extrinsics_from_look_at(views_cartesian[t, j],
                                                       look_at[t, j])

    return extrinsics


def render_with_random_view(
        ply_path: str,
        out_png: str,
        hdri_path: str,  # ← NEW
        width: int = 576,
        height: int = 384,
        intrinsics: np.ndarray = None,
        env_intensity: float = 300000.0):  # ← optional
    import open3d as o3d
    import numpy as np

    # 1) Load mesh
    mesh = o3d.io.read_triangle_mesh(ply_path, enable_post_processing=True)
    mesh.compute_vertex_normals()

    # 2) Save it back out
    out_ply = "test/hey.ply"
    o3d.io.write_triangle_mesh(
        out_ply,
        mesh,
        write_ascii=True,     # or False for binary
        compressed=False,     # or True to gzip-compress
        print_progress=True   # optional
    )
    print(f"Saved re-exported mesh to {out_ply}")

    # 2) Create off-screen renderer and add the mesh
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("mesh", mesh, mat)

    # 3) Random camera
    views = np.zeros((1, 3), dtype=float)
    views[0] = _sample_random_view()  # azimuth, elevation, roll  (your helper)
    extrinsic = np.linalg.inv(np.squeeze(calculate_se3fromAER(views)))

    # 4) Camera intrinsics
    # --- build an Open3D PinholeCameraIntrinsic --------------------------------
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    intr_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    #----Your changes here-----
    #--------------------------
    # --- apply intrinsics + extrinsics to the active camera --------------------
    renderer.setup_camera(intr_o3d, extrinsic)

    # 5) Render & save
    img = renderer.render_to_image()
    o3d.io.write_image(out_png, img)
    print(f"Saved render to {out_png}")


if __name__ == "__main__":
    K = np.array([[256.0, 0.0, 128.0], [0.0, 384.0, 128.0], [0.0, 0.0, 1.0]],
                 dtype=np.float64)
    #K = np.array([[576.0, 0.0, 288.0], [0.0, 576.0, 288.0], [0.0, 0.0, 1.0]],
    #             dtype=np.float64)
    render_with_random_view(
        "/dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/di97nip/Kubric_generation/scn00000/frame_0000.ply",
        "randomview.png",
        "test/frame.hdr",
        width=256,
        height=256,
        intrinsics=K)
