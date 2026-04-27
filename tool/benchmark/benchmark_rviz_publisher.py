#!/usr/bin/env python3
"""Publish benchmark mapping results to RViz for visualization.

Publishes with TRANSIENT_LOCAL (latched) QoS so RViz can connect at any time
after the data is published and still receive all messages.

Usage:
    python3 benchmark_rviz_publisher.py --output_dir /path/to/benchmark_output

When running inside Docker, the container must use --network host so that
ROS2 DDS multicast reaches the host machine's RViz.

Topics published:
    /benchmark/map_a/occupancy_grid       (nav_msgs/OccupancyGrid)   – 2-D top-down map (complete)
    /benchmark/map_a/global_cloud         (sensor_msgs/PointCloud2)  – dense 3-D point cloud from depth images
    /benchmark/map_a/occupied_voxels      (sensor_msgs/PointCloud2)  – raycasted 3-D voxels, coloured by Z
    /benchmark/map_a/trajectory           (nav_msgs/Path)            – bag-A keyframe trajectory
    /benchmark/map_b/localization_traj    (nav_msgs/Path)            – bag-B relocalised in map-A frame
    /benchmark/map_b/ground_truth_traj    (nav_msgs/Path)            – bag-B GT transformed to map-A frame
"""

import argparse
import os
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header


_LATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def _pose_matrix_to_pose_stamped(pose: np.ndarray, frame_id: str, stamp) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.pose.position.x = float(pose[0, 3])
    msg.pose.position.y = float(pose[1, 3])
    msg.pose.position.z = float(pose[2, 3])
    q = R.from_matrix(pose[:3, :3]).as_quat()
    msg.pose.orientation.x = float(q[0])
    msg.pose.orientation.y = float(q[1])
    msg.pose.orientation.z = float(q[2])
    msg.pose.orientation.w = float(q[3])
    return msg


def _poses_to_path(poses_dict: dict, frame_id: str, stamp) -> Path:
    path = Path()
    path.header.frame_id = frame_id
    path.header.stamp = stamp
    for ts in sorted(poses_dict.keys()):
        pose = poses_dict[ts]
        if isinstance(pose, tuple):
            pose = pose[0]
        path.poses.append(_pose_matrix_to_pose_stamped(pose, frame_id, stamp))
    return path


def _occupancy_to_pointcloud2(
    grid: np.ndarray, meta: np.ndarray, stamp, frame_id: str = "world"
) -> PointCloud2:
    """Convert 3-D occupancy grid (grid[ix,iy,iz]==2 → occupied) to PointCloud2.

    meta = [x_origin, y_origin, z_origin, resolution]
    Points are coloured by Z height with matplotlib's 'rainbow' colourmap.
    """
    x_origin, y_origin, z_origin, resolution = (
        float(meta[0]), float(meta[1]), float(meta[2]), float(meta[3])
    )

    occ_idx = np.argwhere(grid == 2)  # (N, 3)
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id

    if occ_idx.size == 0:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        return pc2.create_cloud(header, fields, [])

    xyz = np.column_stack([
        x_origin + occ_idx[:, 0] * resolution,
        y_origin + occ_idx[:, 1] * resolution,
        z_origin + occ_idx[:, 2] * resolution,
    ]).astype(np.float32)

    # Rainbow colour by Z height (vectorised, no Python loop)
    import matplotlib.pyplot as plt
    z = xyz[:, 2]
    z_range = z.max() - z.min()
    t = (z - z.min()) / z_range if z_range > 0 else np.zeros_like(z)
    rgba = plt.cm.rainbow(t)  # (N, 4)
    r = (rgba[:, 0] * 255).astype(np.uint8)
    g = (rgba[:, 1] * 255).astype(np.uint8)
    b = (rgba[:, 2] * 255).astype(np.uint8)
    packed = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)

    dtype = np.dtype([
        ("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)
    ])
    structured = np.zeros(len(xyz), dtype=dtype)
    structured["x"] = xyz[:, 0]
    structured["y"] = xyz[:, 1]
    structured["z"] = xyz[:, 2]
    structured["rgb"] = packed.view(np.float32)

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    return pc2.create_cloud(header, fields, structured)


def save_pcd(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a binary PCD v0.7 file with XYZRGB fields.

    points : (N, 3) float32   – XYZ in metres
    colors : (N,)   uint32    – packed 0x00RRGGBB
    """
    N = len(points)
    rgb_f32 = colors.astype(np.uint32).view(np.float32)

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {N}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {N}\n"
        "DATA binary\n"
    ).encode("ascii")

    # Interleave x, y, z, rgb as packed float32
    data = np.zeros(N, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("rgb", "f4")])
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    data["rgb"] = rgb_f32

    with open(path, "wb") as f:
        f.write(header)
        data.tofile(f)
    print(f"[pcd] Saved {N:,} points to {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def load_pcd(path: str):
    """Read a binary PCD v0.7 file with XYZRGB fields.

    Returns (points, colors) with shapes (N, 3) float32 and (N,) uint32.
    """
    with open(path, "rb") as f:
        # Parse ASCII header until DATA line
        header: dict[str, str] = {}
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            key, _, value = line.partition(" ")
            header[key.upper()] = value.strip()
            if key.upper() == "DATA":
                break

        n = int(header.get("POINTS", 0))
        data_type = header.get("DATA", "binary").lower()
        fields = header.get("FIELDS", "").split()
        sizes = [int(s) for s in header.get("SIZE", "").split()]

        if data_type != "binary":
            raise ValueError(f"load_pcd: only binary PCD is supported, got '{data_type}'")

        row_bytes = sum(sizes)
        raw = np.frombuffer(f.read(n * row_bytes), dtype=np.uint8).reshape(n, row_bytes)

    # Extract x, y, z, rgb by byte offsets derived from header
    offsets: dict[str, int] = {}
    off = 0
    for field, size in zip(fields, sizes):
        offsets[field] = off
        off += size

    def _col(name: str, dtype=np.float32) -> np.ndarray:
        o = offsets[name]
        return raw[:, o: o + 4].copy().view(dtype).reshape(-1)

    points = np.stack([_col("x"), _col("y"), _col("z")], axis=1)
    colors = _col("rgb", dtype=np.uint32) if "rgb" in offsets else np.zeros(n, dtype=np.uint32)
    print(f"[pcd] Loaded {n:,} points from {path}")
    return points, colors


def _voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float):
    """Keep one point per voxel (first-in wins)."""
    if points.size == 0 or voxel_size <= 0:
        return points, colors
    coords = np.floor(points / voxel_size).astype(np.int32)
    # Pack (ix, iy, iz) into a single void key for np.unique
    view = np.ascontiguousarray(coords).view(
        np.dtype((np.void, coords.dtype.itemsize * 3))
    ).reshape(-1)
    _, idx = np.unique(view, return_index=True)
    idx.sort()
    return points[idx], colors[idx]


def _build_map_b_local_cloud(
    map_b_path: str,
    map_a_path: str,
    stamp,
    num_keyframes: int = 15,
    voxel_size: float = 0.05,
    frame_id: str = "world",
    max_depth: float = 5.0,
) -> PointCloud2 | None:
    """Backproject depth images from bag-B keyframes into the map-A world frame.

    Uses relocalization_poses (already in map-A frame) so the resulting cloud
    overlays directly on map A. Points are coloured orange to distinguish from
    the grayscale map-A cloud.

    num_keyframes : how many keyframes to sample evenly along the trajectory.
    """
    import shelve

    reloc_poses = _load_poses(
        os.path.join(map_b_path, "relocalization_poses.npy")
    )
    if not reloc_poses:
        print("[map_b_cloud] No relocalization poses found, skipping")
        return None

    K = np.load(os.path.join(map_a_path, "intrinsics.npy"), allow_pickle=True)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    nav_temp = os.path.join(map_b_path, "nav_temp")
    depths_db = shelve.open(os.path.join(nav_temp, "depths"), flag="r")
    try:
        infra1_db = shelve.open(os.path.join(nav_temp, "infra1_images"), flag="r")
    except Exception:
        infra1_db = None

    # Sample evenly spaced keyframes from the sorted relocalization poses
    ts_all = sorted(reloc_poses.keys())
    n = min(num_keyframes, len(ts_all))
    indices = np.linspace(0, len(ts_all) - 1, n, dtype=int)
    sampled_ts = [ts_all[i] for i in indices]
    print(f"[map_b_cloud] Sampling {n}/{len(ts_all)} keyframes from bag-B localization trajectory")

    # Orange tint: R=255, G=140, B=0 → 0xFF8C00
    ORANGE_R, ORANGE_G, ORANGE_B = 255, 140, 0

    all_pts: list = []
    all_rgb: list = []
    u_grid = v_grid = None

    for ts in sampled_ts:
        key = str(ts)
        if key not in depths_db:
            continue

        depth = depths_db[key]
        T = reloc_poses[ts]            # T_world_camera in map-A frame
        H, W = depth.shape
        valid = (depth > 0.1) & (depth < max_depth)
        if not valid.any():
            continue

        if u_grid is None:
            v_grid, u_grid = np.meshgrid(
                np.arange(H, dtype=np.float32),
                np.arange(W, dtype=np.float32),
                indexing="ij",
            )

        z = depth[valid]
        pts_cam = np.stack(
            [(u_grid[valid] - cx) * z / fx,
             (v_grid[valid] - cy) * z / fy,
             z],
            axis=1,
        )
        pts_world = (T[:3, :3] @ pts_cam.T).T + T[:3, 3]

        # Orange with slight intensity modulation from infra1 for depth cues
        if infra1_db is not None and key in infra1_db:
            intensity = infra1_db[key][valid].astype(np.float32) / 255.0
        else:
            intensity = np.ones(len(z), dtype=np.float32)

        r = (np.clip(ORANGE_R * intensity, 0, 255)).astype(np.uint32)
        g = (np.clip(ORANGE_G * intensity, 0, 255)).astype(np.uint32)
        b = (np.clip(ORANGE_B * intensity, 0, 255)).astype(np.uint32)
        rgb = (r << 16) | (g << 8) | b

        all_pts.append(pts_world.astype(np.float32))
        all_rgb.append(rgb)

    depths_db.close()
    if infra1_db is not None:
        infra1_db.close()

    if not all_pts:
        return None

    pts = np.vstack(all_pts)
    rgb = np.concatenate(all_rgb)
    pts, rgb = _voxel_downsample(pts, rgb, voxel_size)
    print(f"[map_b_cloud] {len(pts):,} points from {n} bag-B keyframes (voxel={voxel_size}m)")
    return _pts_to_pointcloud2(pts, rgb, stamp, frame_id)


def _pts_to_pointcloud2(points: np.ndarray, colors: np.ndarray, stamp, frame_id: str) -> PointCloud2:
    dtype = np.dtype([
        ("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)
    ])
    structured = np.zeros(len(points), dtype=dtype)
    structured["x"] = points[:, 0]
    structured["y"] = points[:, 1]
    structured["z"] = points[:, 2]
    structured["rgb"] = colors.astype(np.uint32).view(np.float32)
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return pc2.create_cloud(header, fields, structured)


def _build_global_pointcloud(map_a_path: str, stamp, voxel_size: float = 0.05,
                              frame_id: str = "world", max_depth: float = 5.0,
                              pcd_path: str | None = None):
    """Return a PointCloud2 of the complete map-A scene.

    If pcd_path exists on disk: load the cached PCD (fast).
    If pcd_path is given but missing: build from depth images, save PCD, then return.
    If pcd_path is None: build from depth images without saving.
    """
    # --- fast path: load from cached PCD ---
    if pcd_path and os.path.exists(pcd_path):
        pts, rgb = load_pcd(pcd_path)
        return _pts_to_pointcloud2(pts.astype(np.float32), rgb, stamp, frame_id)

    # --- slow path: build from depth images ---
    import shelve

    poses = _load_poses(os.path.join(map_a_path, "poses.npy"))
    K = np.load(os.path.join(map_a_path, "intrinsics.npy"), allow_pickle=True)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    depths_db = shelve.open(os.path.join(map_a_path, "depths"), flag="r")
    try:
        infra1_db = shelve.open(os.path.join(map_a_path, "infra1_images"), flag="r")
    except Exception:
        infra1_db = None

    all_pts: list = []
    all_rgb: list = []
    ts_sorted = sorted(poses.keys())
    print(f"[global_cloud] Building from {len(ts_sorted)} keyframes …")

    u_grid = v_grid = None
    for i, ts in enumerate(ts_sorted):
        key = str(ts)
        if key not in depths_db:
            continue

        depth = depths_db[key]          # (H, W) float32, metres
        T = poses[ts]                   # T_world_camera (4×4)
        H, W = depth.shape
        valid = (depth > 0.1) & (depth < max_depth)
        if not valid.any():
            continue

        if u_grid is None:
            v_grid, u_grid = np.meshgrid(
                np.arange(H, dtype=np.float32),
                np.arange(W, dtype=np.float32),
                indexing="ij",
            )

        z = depth[valid]
        pts_cam = np.stack([(u_grid[valid] - cx) * z / fx,
                             (v_grid[valid] - cy) * z / fy,
                             z], axis=1)
        pts_world = (T[:3, :3] @ pts_cam.T).T + T[:3, 3]

        if infra1_db is not None and key in infra1_db:
            gray = infra1_db[key][valid].astype(np.uint32)
        else:
            gray = np.full(len(z), 180, dtype=np.uint32)

        all_pts.append(pts_world.astype(np.float32))
        all_rgb.append((gray << 16) | (gray << 8) | gray)

        if (i + 1) % 50 == 0 or (i + 1) == len(ts_sorted):
            print(f"[global_cloud]   {i + 1}/{len(ts_sorted)} frames processed")

    depths_db.close()
    if infra1_db is not None:
        infra1_db.close()

    if not all_pts:
        return None

    pts = np.vstack(all_pts)
    rgb = np.concatenate(all_rgb)
    pts, rgb = _voxel_downsample(pts, rgb, voxel_size)
    print(f"[global_cloud] {len(pts):,} points after {voxel_size}m voxel downsample")

    if pcd_path:
        save_pcd(pcd_path, pts, rgb)

    return _pts_to_pointcloud2(pts, rgb, stamp, frame_id)


def _occupancy_to_grid2d(
    grid: np.ndarray, meta: np.ndarray, stamp, frame_id: str = "world"
) -> OccupancyGrid:
    """Project 3-D occupancy grid onto the XY plane and return nav_msgs/OccupancyGrid.

    Projection rule: a cell is occupied if any Z-layer is occupied (2);
    otherwise free (1) if any layer is free; otherwise unknown (-1).
    OccupancyGrid convention: 0=free, 100=occupied, -1=unknown.
    """
    x_origin, y_origin, _, resolution = (
        float(meta[0]), float(meta[1]), float(meta[2]), float(meta[3])
    )
    nx, ny, _ = grid.shape

    # Top-down projection: max value along Z axis → shape (nx, ny)
    xy = np.max(grid, axis=2)

    # Remap to ROS OccupancyGrid values
    ros_data = np.full(nx * ny, -1, dtype=np.int8)
    flat = xy.T.flatten()          # row-major (y outer, x inner) for OccupancyGrid
    ros_data[flat == 1] = 0        # free
    ros_data[flat == 2] = 100      # occupied

    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.info.resolution = resolution
    msg.info.width = nx            # cells along X
    msg.info.height = ny           # cells along Y
    msg.info.origin.position.x = x_origin
    msg.info.origin.position.y = y_origin
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.w = 1.0
    msg.data = ros_data.tolist()
    return msg


def _load_poses(path: str) -> dict:
    poses = np.load(path, allow_pickle=True).item()
    # Unwrap legacy (matrix, velocity) tuples
    return {
        ts: (v[0] if isinstance(v, tuple) else v)
        for ts, v in poses.items()
    }


class BenchmarkRvizPublisher(Node):
    def __init__(self, output_dir: str, map_a_path: str,
                 pcd_path: str | None = None, num_map_b_keyframes: int = 15):
        super().__init__("benchmark_rviz_publisher")

        map_b_path = os.path.join(output_dir, "benchmark_map_b")

        self._grid2d_pub = self.create_publisher(
            OccupancyGrid, "/benchmark/map_a/occupancy_grid", _LATCHED_QOS
        )
        self._global_cloud_pub = self.create_publisher(
            PointCloud2, "/benchmark/map_a/global_cloud", _LATCHED_QOS
        )
        self._occ_pub = self.create_publisher(
            PointCloud2, "/benchmark/map_a/occupied_voxels", _LATCHED_QOS
        )
        self._map_a_traj_pub = self.create_publisher(
            Path, "/benchmark/map_a/trajectory", _LATCHED_QOS
        )
        self._loc_traj_pub = self.create_publisher(
            Path, "/benchmark/map_b/localization_traj", _LATCHED_QOS
        )
        self._gt_traj_pub = self.create_publisher(
            Path, "/benchmark/map_b/ground_truth_traj", _LATCHED_QOS
        )
        self._map_b_cloud_pub = self.create_publisher(
            PointCloud2, "/benchmark/map_b/local_cloud", _LATCHED_QOS
        )

        self.get_logger().info(f"Loading benchmark data from: {output_dir}")
        stamp = self.get_clock().now().to_msg()

        # 2D occupancy grid and raycasted 3D voxels (map A)
        self._grid2d_msg, self._occ_cloud = self._load_occupancy(map_a_path, stamp)

        # Dense 3D point cloud – loads from PCD cache if available, else builds from depths
        self._global_cloud = _build_global_pointcloud(map_a_path, stamp, pcd_path=pcd_path)

        # Map A keyframe trajectory
        self._map_a_path_msg = self._load_path(
            os.path.join(map_a_path, "poses.npy"), "map A keyframe", stamp
        )

        # Map B relocalization trajectory — already in map A's world frame
        self._loc_path_msg = self._load_path(
            os.path.join(map_b_path, "relocalization_poses.npy"),
            "map B relocalization", stamp,
        )

        # Map B ground truth trajectory transformed to map A frame
        self._gt_path_msg = self._load_ground_truth_path(
            map_b_path, output_dir, stamp
        )

        # Map B local point cloud from sampled keyframes (orange, in map-A frame)
        self._map_b_cloud = _build_map_b_local_cloud(
            map_b_path, map_a_path, stamp, num_keyframes=num_map_b_keyframes
        )

        self._publish_all()
        # Republish every 5 s so late-joining RViz instances still see data
        self.create_timer(5.0, self._publish_all)
        self.get_logger().info(
            "Publishing benchmark visualisation. Press Ctrl+C to stop.\n"
            "  /benchmark/map_a/occupancy_grid      – Map A 2-D floor plan (complete)\n"
            "  /benchmark/map_a/global_cloud        – Map A dense 3-D point cloud (grey)\n"
            "  /benchmark/map_a/occupied_voxels     – Map A raycasted 3-D voxels\n"
            "  /benchmark/map_a/trajectory          – Map A trajectory (blue)\n"
            "  /benchmark/map_b/local_cloud         – Bag-B sampled keyframe clouds (orange)\n"
            "  /benchmark/map_b/localization_traj   – Bag-B in map-A frame (red)\n"
            "  /benchmark/map_b/ground_truth_traj   – Bag-B GT in map-A frame (green)"
        )

    # ------------------------------------------------------------------
    def _load_occupancy(self, map_a_path: str, stamp):
        grid_file = os.path.join(map_a_path, "occupancy_grid.npy")
        meta_file = os.path.join(map_a_path, "occupancy_meta.npy")
        if not (os.path.exists(grid_file) and os.path.exists(meta_file)):
            self.get_logger().warn(f"Occupancy grid not found in {map_a_path}")
            return None, None
        grid = np.load(grid_file, allow_pickle=True)
        meta = np.load(meta_file, allow_pickle=True)
        nx, ny, nz = grid.shape
        self.get_logger().info(
            f"Loaded occupancy grid {grid.shape} (res={float(meta[3])}m): "
            f"{int((grid == 2).sum())} occupied, {int((grid == 1).sum())} free voxels"
        )
        return _occupancy_to_grid2d(grid, meta, stamp), _occupancy_to_pointcloud2(grid, meta, stamp)

    def _load_path(self, poses_file: str, label: str, stamp) -> Path | None:
        if not os.path.exists(poses_file):
            self.get_logger().warn(f"{label} poses not found: {poses_file}")
            return None
        poses = _load_poses(poses_file)
        self.get_logger().info(f"Loaded {label} trajectory: {len(poses)} poses")
        return _poses_to_path(poses, "world", stamp)

    def _load_ground_truth_path(self, map_b_path: str, output_dir: str, stamp) -> Path | None:
        poses_b_file = os.path.join(map_b_path, "poses.npy")
        transform_file = os.path.join(output_dir, "transformation_matrix.npy")
        if not os.path.exists(poses_b_file):
            self.get_logger().warn(f"Map B poses not found: {poses_b_file}")
            return None
        if not os.path.exists(transform_file):
            self.get_logger().warn(
                f"Transformation matrix not found: {transform_file} — "
                "ground truth trajectory will not be shown"
            )
            return None
        poses_b = _load_poses(poses_b_file)
        # T_a_to_b maps localization poses (map A frame) → ground truth (map B frame).
        # Invert to bring map B GT poses into map A frame for a common view.
        T_a_to_b = np.load(transform_file, allow_pickle=True)
        T_b_to_a = np.linalg.inv(T_a_to_b)
        poses_b_in_a = {ts: T_b_to_a @ pose for ts, pose in poses_b.items()}
        self.get_logger().info(
            f"Loaded map B ground truth trajectory: {len(poses_b)} poses "
            "(transformed to map A frame)"
        )
        return _poses_to_path(poses_b_in_a, "world", stamp)

    # ------------------------------------------------------------------
    def _publish_all(self):
        stamp = self.get_clock().now().to_msg()
        if self._grid2d_msg is not None:
            self._grid2d_msg.header.stamp = stamp
            self._grid2d_pub.publish(self._grid2d_msg)
        if self._global_cloud is not None:
            self._global_cloud.header.stamp = stamp
            self._global_cloud_pub.publish(self._global_cloud)
        if self._occ_cloud is not None:
            self._occ_cloud.header.stamp = stamp
            self._occ_pub.publish(self._occ_cloud)
        if self._map_a_path_msg is not None:
            self._map_a_path_msg.header.stamp = stamp
            self._map_a_traj_pub.publish(self._map_a_path_msg)
        if self._loc_path_msg is not None:
            self._loc_path_msg.header.stamp = stamp
            self._loc_traj_pub.publish(self._loc_path_msg)
        if self._gt_path_msg is not None:
            self._gt_path_msg.header.stamp = stamp
            self._gt_traj_pub.publish(self._gt_path_msg)
        if self._map_b_cloud is not None:
            self._map_b_cloud.header.stamp = stamp
            self._map_b_cloud_pub.publish(self._map_b_cloud)


def main():
    parser = argparse.ArgumentParser(
        description="Publish benchmark_mapping results to RViz"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Benchmark output directory"
    )
    parser.add_argument(
        "--map_a_path",
        default=None,
        help=(
            "Map A directory (default: <output_dir>/benchmark_map_a). "
            "Override when --map_a_path was used in benchmark_mapping.py."
        ),
    )
    parser.add_argument(
        "--num_map_b_keyframes",
        type=int,
        default=15,
        help="Number of keyframes sampled evenly from bag-B localization trajectory "
             "to build the orange local cloud (default: 15).",
    )
    parser.add_argument(
        "--pcd_path",
        default=None,
        help=(
            "Path to a .pcd cache file for the dense 3-D point cloud. "
            "Defaults to <map_a_path>/global_cloud.pcd. "
            "If the file exists it is loaded directly (fast). "
            "If it does not exist the cloud is built from depth images and "
            "saved here for future runs."
        ),
    )
    args = parser.parse_args()

    map_a_path = args.map_a_path or os.path.join(args.output_dir, "benchmark_map_a")

    if not os.path.isdir(args.output_dir):
        print(f"Error: output_dir not found: {args.output_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(map_a_path):
        print(f"Error: map_a_path not found: {map_a_path}", file=sys.stderr)
        sys.exit(1)

    pcd_path = args.pcd_path or os.path.join(map_a_path, "global_cloud.pcd")

    rclpy.init()
    node = BenchmarkRvizPublisher(
        args.output_dir, map_a_path,
        pcd_path=pcd_path,
        num_map_b_keyframes=args.num_map_b_keyframes,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
