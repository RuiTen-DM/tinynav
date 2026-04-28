#!/usr/bin/env python3
"""Publish a TinyNav map package to RViz and export it as PCD.

Example:
    python3 tool/map_package_publisher.py \
        --map-path /home/ruiten/tinynav/output/map_0743_with_ground_plane_constraint

    python3 tool/map_package_publisher.py \
        --map-path /home/ruiten/tinynav/output/map_0743_with_ground_plane_constraint \
        --export-pcd-only
"""

from __future__ import annotations

import argparse
import os
import shelve
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_MAP_PATH = "/home/ruiten/tinynav/output/map_0743_with_ground_plane_constraint"
FRAME_ID = "world"


def load_poses(path: Path) -> dict[int, np.ndarray]:
    poses = np.load(path, allow_pickle=True).item()
    return {int(ts): (pose[0] if isinstance(pose, tuple) else pose) for ts, pose in poses.items()}


def pack_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (
        (r.astype(np.uint32) << 16)
        | (g.astype(np.uint32) << 8)
        | b.astype(np.uint32)
    )


def z_to_rgb(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0,), dtype=np.uint32)
    z = points[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    span = max(z_max - z_min, 1e-6)
    t = np.clip((z - z_min) / span, 0.0, 1.0)
    r = np.clip(255.0 * np.maximum(0.0, 1.5 - np.abs(4.0 * t - 3.0)), 0, 255)
    g = np.clip(255.0 * np.maximum(0.0, 1.5 - np.abs(4.0 * t - 2.0)), 0, 255)
    b = np.clip(255.0 * np.maximum(0.0, 1.5 - np.abs(4.0 * t - 1.0)), 0, 255)
    return pack_rgb(r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8))


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float):
    if points.size == 0 or voxel_size <= 0.0:
        return points.reshape(-1, 3), colors.reshape(-1)
    coords = np.floor(points / voxel_size).astype(np.int32)
    coord_view = (
        np.ascontiguousarray(coords)
        .view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1])))
        .reshape(-1)
    )
    _, unique_idx = np.unique(coord_view, return_index=True)
    unique_idx.sort()
    return points[unique_idx], colors[unique_idx]


def open_shelf(map_path: Path, stem: str):
    return shelve.open(str(map_path / stem), flag="r")


def shelf_exists(map_path: Path, stem: str) -> bool:
    return any((map_path / f"{stem}{suffix}").exists() for suffix in ("", ".db", ".dat", ".dir", ".bak"))


def make_uv_grid(depth_shape: tuple[int, int], stride: int):
    h, w = depth_shape
    u_coords = np.arange(0, w, stride, dtype=np.float32)
    v_coords = np.arange(0, h, stride, dtype=np.float32)
    return np.meshgrid(u_coords, v_coords)


def depth_to_world_cloud(
    depth: np.ndarray,
    gray_image: np.ndarray | None,
    pose: np.ndarray,
    K: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    stride: int,
    min_depth: float,
    max_depth: float,
):
    sampled_depth = depth[::stride, ::stride].astype(np.float32, copy=False)
    valid = np.isfinite(sampled_depth) & (sampled_depth >= min_depth) & (sampled_depth <= max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.uint32)

    z = sampled_depth[valid]
    u = u_grid[valid]
    v = v_grid[valid]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    points_cam = np.column_stack((x, y, z)).astype(np.float32)
    points_world = (points_cam @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)

    if gray_image is not None and gray_image.shape[:2] == depth.shape:
        gray = gray_image[::stride, ::stride][valid].astype(np.uint32)
    else:
        gray = np.full(points_world.shape[0], 180, dtype=np.uint32)
    colors = (gray << 16) | (gray << 8) | gray
    return points_world, colors


def build_dense_cloud(
    map_path: Path,
    depth_stride: int,
    keyframe_stride: int,
    min_depth: float,
    max_depth: float,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    poses = load_poses(map_path / "poses.npy")
    K = np.load(map_path / "intrinsics.npy", allow_pickle=True).astype(np.float32)
    depths_db = open_shelf(map_path, "depths")
    try:
        infra1_db = open_shelf(map_path, "infra1_images")
    except Exception:
        infra1_db = None

    all_points = []
    all_colors = []
    sorted_ts = sorted(poses.keys())[::max(1, keyframe_stride)]
    uv_key = None
    u_grid = v_grid = None
    print(f"[dense] Building cloud from {len(sorted_ts)} keyframes in {map_path}")

    try:
        for idx, timestamp in enumerate(sorted_ts):
            key = str(timestamp)
            if key not in depths_db:
                continue
            depth = depths_db[key]
            if uv_key != depth.shape:
                u_grid, v_grid = make_uv_grid(depth.shape, depth_stride)
                uv_key = depth.shape
            gray = infra1_db[key] if infra1_db is not None and key in infra1_db else None
            points, colors = depth_to_world_cloud(
                depth,
                gray,
                poses[timestamp],
                K,
                u_grid,
                v_grid,
                depth_stride,
                min_depth,
                max_depth,
            )
            points, colors = voxel_downsample(points, colors, voxel_size)
            if points.size:
                all_points.append(points)
                all_colors.append(colors)
            if (idx + 1) % 25 == 0 or idx + 1 == len(sorted_ts):
                print(f"[dense]   {idx + 1}/{len(sorted_ts)} keyframes processed")
    finally:
        depths_db.close()
        if infra1_db is not None:
            infra1_db.close()

    if not all_points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.uint32)

    points = np.vstack(all_points).astype(np.float32, copy=False)
    colors = np.concatenate(all_colors).astype(np.uint32, copy=False)
    points, colors = voxel_downsample(points, colors, voxel_size)
    print(f"[dense] {points.shape[0]:,} points after {voxel_size:.3f} m voxel downsample")
    return points, colors


def build_occupied_voxel_cloud(map_path: Path) -> tuple[np.ndarray, np.ndarray]:
    grid_path = map_path / "occupancy_grid.npy"
    meta_path = map_path / "occupancy_meta.npy"
    if not grid_path.exists() or not meta_path.exists():
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.uint32)
    grid = np.load(grid_path, allow_pickle=True)
    meta = np.load(meta_path, allow_pickle=True).astype(np.float32)
    occupied = np.argwhere(grid == 2)
    if occupied.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.uint32)
    origin = meta[:3]
    resolution = float(meta[3])
    points = (origin + occupied.astype(np.float32) * resolution).astype(np.float32)
    colors = z_to_rgb(points)
    print(f"[occupancy] {points.shape[0]:,} occupied voxels")
    return points, colors


def save_pcd(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = points.astype(np.float32, copy=False)
    colors = colors.astype(np.uint32, copy=False)
    rgb_as_float = colors.view(np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\n"
        "DATA binary\n"
    ).encode("ascii")
    data = np.zeros(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("rgb", "f4")])
    if len(points):
        data["x"] = points[:, 0]
        data["y"] = points[:, 1]
        data["z"] = points[:, 2]
        data["rgb"] = rgb_as_float
    with path.open("wb") as f:
        f.write(header)
        data.tofile(f)
    print(f"[pcd] Saved {len(points):,} points to {path}")


def load_pcd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header: dict[str, str] = {}
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(" ")
            header[key.upper()] = value.strip()
            if key.upper() == "DATA":
                break

        if header.get("DATA", "").lower() != "binary":
            raise ValueError(f"Only binary PCD is supported: {path}")
        n = int(header.get("POINTS", "0"))
        fields = header.get("FIELDS", "").split()
        sizes = [int(v) for v in header.get("SIZE", "").split()]
        row_bytes = sum(sizes)
        raw = np.frombuffer(f.read(n * row_bytes), dtype=np.uint8).reshape(n, row_bytes)

    offsets = {}
    offset = 0
    for field, size in zip(fields, sizes):
        offsets[field] = offset
        offset += size

    def col(name: str, dtype=np.float32):
        start = offsets[name]
        return raw[:, start:start + 4].copy().view(dtype).reshape(-1)

    points = np.column_stack((col("x"), col("y"), col("z"))).astype(np.float32)
    colors = col("rgb", np.uint32).astype(np.uint32) if "rgb" in offsets else z_to_rgb(points)
    print(f"[pcd] Loaded {len(points):,} points from {path}")
    return points, colors


def load_or_build_dense_cloud(args) -> tuple[np.ndarray, np.ndarray]:
    map_path = Path(args.map_path)
    pcd_path = Path(args.pcd_path) if args.pcd_path else map_path / "global_cloud.pcd"
    if pcd_path.exists() and not args.rebuild_pcd:
        return load_pcd(pcd_path)
    points, colors = build_dense_cloud(
        map_path,
        depth_stride=args.depth_stride,
        keyframe_stride=args.keyframe_stride,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        voxel_size=args.voxel_size,
    )
    save_pcd(pcd_path, points, colors)
    return points, colors


def pose_to_stamped(pose: np.ndarray, stamp):
    from geometry_msgs.msg import PoseStamped

    msg = PoseStamped()
    msg.header.frame_id = FRAME_ID
    msg.header.stamp = stamp
    msg.pose.position.x = float(pose[0, 3])
    msg.pose.position.y = float(pose[1, 3])
    msg.pose.position.z = float(pose[2, 3])
    quat = R.from_matrix(pose[:3, :3]).as_quat()
    msg.pose.orientation.x = float(quat[0])
    msg.pose.orientation.y = float(quat[1])
    msg.pose.orientation.z = float(quat[2])
    msg.pose.orientation.w = float(quat[3])
    return msg


def make_path_msg(map_path: Path, stamp):
    from nav_msgs.msg import Path

    path = Path()
    path.header.frame_id = FRAME_ID
    path.header.stamp = stamp
    poses = load_poses(map_path / "poses.npy")
    for timestamp in sorted(poses.keys()):
        path.poses.append(pose_to_stamped(poses[timestamp], stamp))
    return path


def make_cloud_msg(points: np.ndarray, colors: np.ndarray, stamp):
    import sensor_msgs_py.point_cloud2 as pc2
    from sensor_msgs.msg import PointField
    from std_msgs.msg import Header

    dtype = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)])
    structured = np.zeros(points.shape[0], dtype=dtype)
    if points.size:
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
    header.frame_id = FRAME_ID
    header.stamp = stamp
    return pc2.create_cloud(header, fields, structured)


def make_occupancy_grid_msg(map_path: Path, stamp):
    from nav_msgs.msg import OccupancyGrid

    grid_path = map_path / "occupancy_grid.npy"
    meta_path = map_path / "occupancy_meta.npy"
    if not grid_path.exists() or not meta_path.exists():
        return None
    grid = np.load(grid_path, allow_pickle=True)
    meta = np.load(meta_path, allow_pickle=True).astype(np.float32)
    xy = np.max(grid, axis=2)
    nx, ny = xy.shape
    flat = xy.T.flatten()
    data = np.full(flat.shape, -1, dtype=np.int8)
    data[flat == 1] = 0
    data[flat == 2] = 100

    msg = OccupancyGrid()
    msg.header.frame_id = FRAME_ID
    msg.header.stamp = stamp
    msg.info.resolution = float(meta[3])
    msg.info.width = int(nx)
    msg.info.height = int(ny)
    msg.info.origin.position.x = float(meta[0])
    msg.info.origin.position.y = float(meta[1])
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.w = 1.0
    msg.data = data.tolist()
    return msg


def make_identity_tf(stamp):
    from geometry_msgs.msg import TransformStamped

    msg = TransformStamped()
    msg.header.frame_id = FRAME_ID
    msg.header.stamp = stamp
    msg.child_frame_id = "map_origin"
    msg.transform.rotation.w = 1.0
    return msg


def publish_to_rviz(args, dense_points: np.ndarray, dense_colors: np.ndarray) -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from nav_msgs.msg import OccupancyGrid, Path as PathMsg
    from sensor_msgs.msg import PointCloud2
    from tf2_ros import StaticTransformBroadcaster

    latched_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    class MapPackagePublisher(Node):
        def __init__(self):
            super().__init__("map_package_publisher")
            map_path = Path(args.map_path)
            stamp = self.get_clock().now().to_msg()

            occupied_points, occupied_colors = build_occupied_voxel_cloud(map_path)
            self.dense_msg = make_cloud_msg(dense_points, dense_colors, stamp)
            self.occupied_msg = make_cloud_msg(occupied_points, occupied_colors, stamp)
            self.occupancy_msg = make_occupancy_grid_msg(map_path, stamp)
            self.path_msg = make_path_msg(map_path, stamp)

            self.dense_pub = self.create_publisher(PointCloud2, "/tinynav/map/global_cloud", latched_qos)
            self.occupied_pub = self.create_publisher(PointCloud2, "/tinynav/map/occupied_voxels", latched_qos)
            self.occupancy_pub = self.create_publisher(OccupancyGrid, "/tinynav/map/occupancy_grid", latched_qos)
            self.path_pub = self.create_publisher(PathMsg, "/tinynav/map/keyframes", latched_qos)
            self.tf_broadcaster = StaticTransformBroadcaster(self)

            self.publish_all()
            self.create_timer(args.publish_period, self.publish_all)
            self.get_logger().info(
                "Publishing TinyNav map package to RViz:\n"
                "  /tinynav/map/global_cloud\n"
                "  /tinynav/map/occupied_voxels\n"
                "  /tinynav/map/occupancy_grid\n"
                "  /tinynav/map/keyframes"
            )

        def publish_all(self):
            stamp = self.get_clock().now().to_msg()
            self.dense_msg.header.stamp = stamp
            self.occupied_msg.header.stamp = stamp
            self.path_msg.header.stamp = stamp
            for pose in self.path_msg.poses:
                pose.header.stamp = stamp
            if self.occupancy_msg is not None:
                self.occupancy_msg.header.stamp = stamp
                self.occupancy_pub.publish(self.occupancy_msg)
            self.dense_pub.publish(self.dense_msg)
            self.occupied_pub.publish(self.occupied_msg)
            self.path_pub.publish(self.path_msg)
            self.tf_broadcaster.sendTransform(make_identity_tf(stamp))

    log_dir = Path(args.map_path) / "ros_logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        log_dir = Path("/tmp/tinynav_map_package_ros_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_LOG_DIR", str(log_dir))

    rclpy.init()
    node = MapPackagePublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish a TinyNav map package to RViz and export a dense PCD cloud."
    )
    parser.add_argument("--map-path", default=DEFAULT_MAP_PATH, help="TinyNav map directory.")
    parser.add_argument("--pcd-path", default=None, help="PCD output/cache path. Default: <map-path>/global_cloud.pcd")
    parser.add_argument("--export-pcd-only", action="store_true", help="Build or load the PCD and exit without ROS publishing.")
    parser.add_argument("--rebuild-pcd", action="store_true", help="Ignore an existing PCD cache and rebuild from depths.db.")
    parser.add_argument("--depth-stride", type=int, default=4, help="Pixel stride used when backprojecting depth images.")
    parser.add_argument("--keyframe-stride", type=int, default=1, help="Use every Nth keyframe when building the dense PCD.")
    parser.add_argument("--min-depth", type=float, default=0.15, help="Minimum accepted depth in metres.")
    parser.add_argument("--max-depth", type=float, default=8.0, help="Maximum accepted depth in metres.")
    parser.add_argument("--voxel-size", type=float, default=0.03, help="Voxel downsample size in metres.")
    parser.add_argument("--publish-period", type=float, default=5.0, help="Republish period in seconds.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args, ros_args = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if ros_args:
        print(f"[warn] Ignoring ROS/unrecognized args: {' '.join(ros_args)}")

    map_path = Path(args.map_path)
    required = ["poses.npy", "intrinsics.npy"]
    missing = [name for name in required if not (map_path / name).exists()]
    if not shelf_exists(map_path, "depths"):
        missing.append("depths shelf")
    if missing:
        print(f"Missing required map files in {map_path}: {', '.join(missing)}", file=sys.stderr)
        return 2

    dense_points, dense_colors = load_or_build_dense_cloud(args)
    if args.export_pcd_only:
        return 0
    publish_to_rviz(args, dense_points, dense_colors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
