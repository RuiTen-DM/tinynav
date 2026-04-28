import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool, Float32
import numpy as np

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from tinynav.core.math_utils import matrix_to_quat, msg2np, estimate_pose, tf2np, depth_to_cloud
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2
from codetiming import Timer
import os
import argparse
import sys

from tinynav.tinynav_cpp_bind import pose_graph_solve
from tinynav.core.models_trt import LightGlueTRT, Dinov2TRT, SuperPointTRT
from tinynav.core.planning_node import run_raycasting_loopy
import logging
import asyncio
import shelve
from tqdm import tqdm
import einops
from tf2_msgs.msg import TFMessage
from typing import Dict, Optional
import time

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped,Point
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import distance_transform_edt

from rclpy.executors import SingleThreadedExecutor
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from rosgraph_msgs.msg import Clock
from rclpy.serialization import deserialize_message



logger = logging.getLogger(__name__)


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def _rotation_from_two_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = _normalize_vector(src.astype(np.float64))
    dst = _normalize_vector(dst.astype(np.float64))
    cross = np.cross(src, dst)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if cross_norm < 1e-8:
        if dot > 0.0:
            return np.eye(3)
        axis = np.cross(src, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0]))
        axis = _normalize_vector(axis)
        return R.from_rotvec(np.pi * axis).as_matrix()
    axis = cross / cross_norm
    angle = np.arctan2(cross_norm, dot)
    return R.from_rotvec(angle * axis).as_matrix()


def _fit_plane_pca(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = _normalize_vector(vh[-1])
    offset = -float(np.dot(normal, centroid))
    return normal.astype(np.float64), offset, centroid.astype(np.float64)


def fit_plane_ransac_pca(
    points: np.ndarray,
    distance_threshold: float,
    max_iterations: int,
    min_inliers: int,
    rng_seed: int = 0,
) -> Optional[dict]:
    if points.shape[0] < max(3, min_inliers):
        return None

    points = points.astype(np.float64, copy=False)
    rng = np.random.default_rng(rng_seed)
    best_inliers = None
    best_count = 0
    best_median_error = np.inf
    point_count = points.shape[0]

    for _ in range(max_iterations):
        sample_idx = rng.choice(point_count, size=3, replace=False)
        p0, p1, p2 = points[sample_idx]
        normal = np.cross(p1 - p0, p2 - p0)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-8:
            continue
        normal = normal / normal_norm
        offset = -float(np.dot(normal, p0))
        distances = np.abs(points @ normal + offset)
        inliers = distances < distance_threshold
        count = int(np.count_nonzero(inliers))
        if count < min_inliers:
            continue
        median_error = float(np.median(distances[inliers]))
        if count > best_count or (count == best_count and median_error < best_median_error):
            best_inliers = inliers
            best_count = count
            best_median_error = median_error

    if best_inliers is None:
        return None

    inlier_points = points[best_inliers]
    normal, offset, centroid = _fit_plane_pca(inlier_points)
    distances = np.abs(points @ normal + offset)
    refined_inliers = distances < distance_threshold
    refined_count = int(np.count_nonzero(refined_inliers))
    if refined_count >= min_inliers:
        inlier_points = points[refined_inliers]
        normal, offset, centroid = _fit_plane_pca(inlier_points)
        best_inliers = refined_inliers
        best_count = refined_count

    return {
        "normal": normal.astype(np.float32),
        "offset": float(offset),
        "centroid": centroid.astype(np.float32),
        "inlier_count": best_count,
        "inlier_ratio": float(best_count / point_count),
    }


def sample_ground_candidates_from_depth(
    depth: np.ndarray,
    K: np.ndarray,
    step: int = 12,
    max_distance: float = 8.0,
    min_distance: float = 0.2,
    min_camera_y: float = 0.05,
    min_v_fraction: float = 0.40,
    max_points: int = 5000,
) -> np.ndarray:
    h, w = depth.shape
    v_start = int(h * min_v_fraction)
    us = np.arange(0, w, step, dtype=np.float32)
    vs = np.arange(v_start, h, step, dtype=np.float32)
    if us.size == 0 or vs.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    uu, vv = np.meshgrid(us, vs)
    z = depth[vv.astype(np.int32), uu.astype(np.int32)].astype(np.float32)
    valid = np.isfinite(z) & (z >= min_distance) & (z <= max_distance)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    z = z[valid]
    uu = uu[valid]
    vv = vv[valid]
    x = (uu - K[0, 2]) * z / K[0, 0]
    y = (vv - K[1, 2]) * z / K[1, 1]
    points = np.stack((x, y, z), axis=1)
    points = points[points[:, 1] > min_camera_y]
    if points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int32)
        points = points[indices]
    return points.astype(np.float32)


def estimate_ground_plane_from_depth(depth: np.ndarray, K: np.ndarray) -> Optional[dict]:
    points = sample_ground_candidates_from_depth(depth, K)
    plane = fit_plane_ransac_pca(
        points,
        distance_threshold=0.04,
        max_iterations=96,
        min_inliers=120,
        rng_seed=17,
    )
    if plane is None or plane["inlier_ratio"] < 0.20:
        return None

    # Camera optical frame has +Y downward; orient the ground normal roughly upward.
    if plane["normal"][1] > 0.0:
        plane["normal"] = -plane["normal"]
        plane["offset"] = -plane["offset"]
    return plane


def _transform_plane_to_world(plane: dict, pose: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    normal = pose[:3, :3] @ plane["normal"].astype(np.float64)
    normal = _normalize_vector(normal)
    centroid = pose[:3, :3] @ plane["centroid"].astype(np.float64) + pose[:3, 3]
    offset = float(plane["offset"] - np.dot(normal, pose[:3, 3]))
    return normal, offset, centroid


def estimate_global_ground_plane(poses: dict, ground_planes: dict) -> Optional[dict]:
    centroids = []
    normals = []
    for timestamp, plane in ground_planes.items():
        if timestamp not in poses:
            continue
        normal, _, centroid = _transform_plane_to_world(plane, poses[timestamp])
        centroids.append(centroid)
        normals.append(normal)

    if len(centroids) < 3:
        return None

    centroids = np.asarray(centroids, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    reference_normal = normals[0]
    for i in range(normals.shape[0]):
        if np.dot(normals[i], reference_normal) < 0.0:
            normals[i] = -normals[i]
    mean_normal = _normalize_vector(np.mean(normals, axis=0))
    if np.linalg.norm(mean_normal) < 1e-8:
        return None

    plane = fit_plane_ransac_pca(
        centroids,
        distance_threshold=0.08,
        max_iterations=128,
        min_inliers=max(3, int(0.35 * len(centroids))),
        rng_seed=23,
    )

    if plane is not None:
        normal = plane["normal"].astype(np.float64)
        if np.dot(normal, mean_normal) < 0.0:
            normal = -normal
        # Straight-line trajectories can make centroid-only plane fitting ill-conditioned.
        if np.dot(normal, mean_normal) > np.cos(np.deg2rad(45.0)):
            mean_normal = _normalize_vector(normal + mean_normal)

    offsets = -(centroids @ mean_normal)
    offset = float(np.median(offsets))
    distances = np.abs(centroids @ mean_normal + offset)
    inliers = distances < 0.08
    if np.count_nonzero(inliers) >= max(3, int(0.35 * len(centroids))):
        offset = float(np.median(offsets[inliers]))
        centroid = np.mean(centroids[inliers], axis=0)
        inlier_count = int(np.count_nonzero(inliers))
    else:
        centroid = np.mean(centroids, axis=0)
        inlier_count = len(centroids)

    plane = {
        "normal": mean_normal.astype(np.float32),
        "offset": offset,
        "centroid": centroid.astype(np.float32),
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_count / len(centroids)),
    }
    plane["source_count"] = len(centroids)
    return plane


def apply_ground_plane_constraint(
    poses: dict,
    ground_planes: dict,
    max_tilt_correction_deg: float = 12.0,
    max_height_correction: float = 0.60,
) -> tuple[dict, dict]:
    global_plane = estimate_global_ground_plane(poses, ground_planes)
    stats = {
        "enabled": True,
        "global_plane": global_plane,
        "corrected_count": 0,
        "skipped_no_plane": 0,
        "skipped_tilt": 0,
        "skipped_height": 0,
        "tilt_correction_deg": [],
        "height_correction": [],
    }
    if global_plane is None:
        stats["enabled"] = False
        return poses, stats

    global_normal = global_plane["normal"].astype(np.float64)
    global_normal = _normalize_vector(global_normal)
    global_offset = float(global_plane["offset"])
    max_tilt_rad = np.deg2rad(max_tilt_correction_deg)
    corrected_poses = {}

    for timestamp, pose in poses.items():
        plane = ground_planes.get(timestamp)
        if plane is None:
            corrected_poses[timestamp] = pose
            stats["skipped_no_plane"] += 1
            continue

        normal_cam = plane["normal"].astype(np.float64)
        offset_cam = float(plane["offset"])
        normal_world = pose[:3, :3] @ normal_cam
        normal_world = _normalize_vector(normal_world)
        if np.dot(normal_world, global_normal) < 0.0:
            normal_cam = -normal_cam
            offset_cam = -offset_cam
            normal_world = -normal_world

        tilt = float(np.arccos(np.clip(np.dot(normal_world, global_normal), -1.0, 1.0)))
        if tilt > max_tilt_rad:
            corrected_poses[timestamp] = pose
            stats["skipped_tilt"] += 1
            continue

        height_delta = float(offset_cam - np.dot(global_normal, pose[:3, 3]) - global_offset)
        if abs(height_delta) > max_height_correction:
            corrected_poses[timestamp] = pose
            stats["skipped_height"] += 1
            continue

        corrected_pose = pose.copy()
        correction_rot = _rotation_from_two_vectors(normal_world, global_normal)
        corrected_pose[:3, :3] = correction_rot @ pose[:3, :3]
        corrected_pose[:3, 3] = pose[:3, 3] + height_delta * global_normal
        corrected_poses[timestamp] = corrected_pose
        stats["corrected_count"] += 1
        stats["tilt_correction_deg"].append(float(np.rad2deg(tilt)))
        stats["height_correction"].append(height_delta)

    stats["tilt_correction_deg"] = np.asarray(stats["tilt_correction_deg"], dtype=np.float32)
    stats["height_correction"] = np.asarray(stats["height_correction"], dtype=np.float32)
    return corrected_poses, stats

def z_value_to_color(z, z_min, z_max):
    color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)
    normalized_z = (z - z_min) / (z_max - z_min)
    if normalized_z < 0.25:
        color.g = normalized_z * 4.0
        color.b = 1.0
    elif normalized_z < 0.5:
        color.g = 1.0
        color.b = 1.0 - (normalized_z - 0.25) * 4.0
    elif normalized_z < 0.75:
        color.r = (normalized_z - 0.5) * 4.0
        color.g = 1.0
    else:
        color.r = 1.0
        color.g = 1.0 - (normalized_z - 0.75) * 4.0
    return color

def merge_local_into_global(global_grid:np.ndarray, global_origin:np.ndarray, local_grid:np.ndarray, local_origin:np.ndarray, resolution:float) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge a local grid into a global grid.
    """
    resolution_half = np.array([resolution / 2.0, resolution / 2.0, resolution / 2.0], dtype=np.float32)
    local_origin_offset = ((local_origin - global_origin + resolution_half) / resolution).astype(np.int32)
    global_grid[local_origin_offset[0]:local_origin_offset[0] + local_grid.shape[0],
                local_origin_offset[1]:local_origin_offset[1] + local_grid.shape[1],
                local_origin_offset[2]:local_origin_offset[2] + local_grid.shape[2]] += local_grid

    return global_grid, global_origin

def solve_pose_graph(pose_graph_used_pose:dict, relative_pose_constraint:list, max_iteration_num:int = 1024) -> dict:
    """
    Solve the bundle adjustment problem.
    """
    if len(relative_pose_constraint) == 0:
        return pose_graph_used_pose
    min_timestamp = min(pose_graph_used_pose.keys())
    constant_pose_index_dict = { min_timestamp : True }

    relative_pose_constraint = [
        (curr_timestamp, prev_timestamp, T_prev_curr, np.array([10.0, 10.0, 10.0]), np.array([30.0, 30.0, 30.0]))
        for curr_timestamp, prev_timestamp, T_prev_curr in relative_pose_constraint]
    optimized_camera_poses = pose_graph_solve(pose_graph_used_pose, relative_pose_constraint, constant_pose_index_dict, max_iteration_num)
    return {t: optimized_camera_poses[t] for t in sorted(optimized_camera_poses.keys())}

def find_loop(target_embedding:np.ndarray, embeddings:np.ndarray, loop_similarity_threshold:float, loop_top_k:int) -> list[tuple[int, float]]:
    if len(embeddings) == 0:
        return []
    similarity_array = einops.einsum(target_embedding, embeddings, "d, n d -> n")
    top_k_indices = np.argsort(similarity_array, axis = 0)
    loop_list = []
    for idx in top_k_indices:
        if similarity_array[idx] > loop_similarity_threshold:
            loop_list.append((idx, similarity_array[idx]))
    return loop_list[-loop_top_k:]

def generate_occupancy_map(poses, db, K, baseline, resolution = 0.1, step = 100):
    """
        Generate a occupancy grid map from the depth images.
        The occupancy grid map is a 3D grid with the following values:
            0 : Unknown
            1 : Free
            2 : Occupied
    """
    raycast_shape = (100, 100, 20)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    odom_pose_min_position = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    odom_pose_max_position = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    for timestamp, odom_pose in poses.items():
        odom_translation = odom_pose[:3, 3]
        odom_pose_min_position = np.minimum(odom_pose_min_position, odom_translation)
        odom_pose_max_position = np.maximum(odom_pose_max_position, odom_translation)
    odom_pose_min_position = np.floor(odom_pose_min_position / resolution) * resolution
    odom_pose_max_position = np.ceil(odom_pose_max_position / resolution) * resolution
    global_grid_shape = np.ceil(
        (odom_pose_max_position - odom_pose_min_position) / resolution + np.array(raycast_shape)
    ).astype(np.int32)
    print(f"global_grid_shape : {global_grid_shape}")
    global_origin = odom_pose_min_position - 0.5 * np.array(raycast_shape) * resolution
    global_grid = np.zeros(global_grid_shape, dtype=np.float32)

    odom_positions = []
    for timestamp, odom_pose in tqdm(poses.items()):
        depth, _, _, _, _ = db.get_depth_embedding_features_images(timestamp)
        odom_translation = odom_pose[:3, 3]
        local_origin = np.floor(odom_translation / resolution) * resolution - 0.5 * np.array(raycast_shape) * resolution
        local_grid = run_raycasting_loopy(depth, odom_pose, raycast_shape, fx, fy, cx, cy, local_origin, step, resolution, filter_ground = True)
        global_grid, global_origin = merge_local_into_global(global_grid, global_origin, local_grid, local_origin, resolution)
        odom_position = odom_pose[:3, 3]
        odom_positions.append(odom_position)

    voxels = int(np.prod(global_grid_shape))
    print(
        "[generate_occupancy_map] SDF stage params: "
        f"resolution={resolution}, step={step}, "
        f"num_poses={len(odom_positions)}, global_grid_shape={tuple(global_grid_shape.tolist())}, "
        f"global_origin={global_origin.tolist()}, voxels={voxels}"
    )

    # Compute SDF as voxel distance to nearest odom seed using SciPy EDT.
    with Timer(name="sdf_distance_transform_edt", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
        if len(odom_positions) == 0:
            sdf_map = np.full(global_grid_shape, np.inf, dtype=np.float32)
        else:
            seed_mask = np.ones(global_grid_shape, dtype=np.uint8)
            odom_positions_np = np.asarray(odom_positions, dtype=np.float32)
            seed_indices = np.rint((odom_positions_np - global_origin) / resolution).astype(np.int32)
            seed_indices = np.clip(seed_indices, 0, global_grid_shape - 1)
            seed_mask[seed_indices[:, 0], seed_indices[:, 1], seed_indices[:, 2]] = 0
            sdf_map = distance_transform_edt(seed_mask, sampling=(resolution, resolution, resolution)).astype(np.float32)

    # 0 is the unknown.
    grid_type = np.zeros_like(global_grid, dtype=np.uint8)

    grid_type[global_grid > 0] = 2  # Occupied
    grid_type[global_grid < 0] = 1  # Free

    x_y_plane = np.max(grid_type, axis=2)
    x_y_plane_image = np.zeros_like(x_y_plane, dtype=np.float32)
    x_y_plane_image[x_y_plane == 2] = 1.0
    x_y_plane_image[x_y_plane == 1] = 0.5
    x_y_plane_image = (x_y_plane_image * 255).astype(np.uint8)
    return grid_type, global_origin, x_y_plane_image, sdf_map

class IntKeyShelf:
    def __init__(self, filename):
        self.db = shelve.open(filename)

    def __getitem__(self, key: int):
        return self.db[str(key)]

    def __setitem__(self, key: int, value):
        self.db[str(key)] = value

    def __delitem__(self, key: int):
        del self.db[str(key)]

    def __contains__(self, key: int):
        return str(key) in self.db

    def keys(self):
        return [int(k) for k in self.db.keys()]

    def close(self):
        self.db.close()


class OdomPoseRecorder:
    """
    Utility class to record continuous odometry data to disk.
    Saves timestamp-pose pairs for later timestamp-based queries.
    """

    def __init__(self, save_path: str, prefix: str = "poses"):
        self.save_path = save_path
        self.prefix = prefix
        self.file_save_path = os.path.join(save_path, f"{prefix}_continuous_odom.npy")
        self.poses: Dict[int, np.ndarray] = {}  # timestamp_ns -> 4x4 pose matrix

        os.makedirs(save_path, exist_ok=True)

    def record_odometry_msg(self, odom_msg: Odometry) -> None:
        timestamp_ns = int(odom_msg.header.stamp.sec * 1e9) + int(
            odom_msg.header.stamp.nanosec
        )
        # msg2np returns (T_4x4, velocity_3); only the pose matrix is needed here.
        pose_matrix, _ = msg2np(odom_msg)
        self.poses[timestamp_ns] = pose_matrix

    def save_to_disk(self) -> None:
        if not self.poses:
            logger.warning(f"No continuous odom poses to save for {self.prefix}")
            return

        logger.info(f"{self.prefix}: Saved {len(self.poses)} continuous odom poses")
        # Create a copy of the dict for saving to avoid any typing issues
        poses_to_save = dict(self.poses)
        np.save(self.file_save_path, poses_to_save, allow_pickle=True)  # type: ignore

        logger.info(f"Saved {len(self.poses)} poses to {self.file_save_path}")

    def load_from_disk(self) -> bool:
        if not os.path.exists(self.file_save_path):
            logger.warning(f"Pose file not found: {self.file_save_path}")
            return False

        try:
            self.poses = np.load(self.file_save_path, allow_pickle=True).item()
            logger.info(
                f"[PoseRecorder] Loaded {len(self.poses)} poses from {self.file_save_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load poses from {self.file_save_path}: {e}")
            return False

    def clear(self) -> None:
        self.poses.clear()


class TinyNavDB():
    def __init__(self, map_save_path:str, is_scratch:bool = True):
        self.map_save_path = map_save_path
        if is_scratch:
            if os.path.exists(f"{map_save_path}/features.db"):
                os.remove(f"{map_save_path}/features.db")
            if os.path.exists(f"{map_save_path}/infra1_images.db"):
                os.remove(f"{map_save_path}/infra1_images.db")
            if os.path.exists(f"{map_save_path}/depths.db"):
                os.remove(f"{map_save_path}/depths.db")
            if os.path.exists(f"{map_save_path}/rgb_images.db"):
                os.remove(f"{map_save_path}/rgb_images.db")
            if os.path.exists(f"{map_save_path}/embeddings.db"):
                os.remove(f"{map_save_path}/embeddings.db")
        self.features = IntKeyShelf(f"{map_save_path}/features")
        self.embeddings = IntKeyShelf(f"{map_save_path}/embeddings")
        self.infra1_images = IntKeyShelf(f"{map_save_path}/infra1_images")
        self.depths = IntKeyShelf(f"{map_save_path}/depths")
        self.rgb_images = IntKeyShelf(f"{map_save_path}/rgb_images")

    def set_entry(self, key:int, depth=None, embedding=None, features=None, infra1_image=None, rgb_image=None):
        if infra1_image is not None:
            self.infra1_images[key] = infra1_image
        if rgb_image is not None:
            self.rgb_images[key] = rgb_image
        if depth is not None:
            self.depths[key] = depth
        if embedding is not None:
            self.embeddings[key] = embedding
        if features is not None:
            self.features[key] = features

    def get_depth_embedding_features_images(self, key:int):
        rgb_image = self.rgb_images[key] if key in self.rgb_images else None
        return self.depths[key], self.embeddings[key], self.features[key], rgb_image, self.infra1_images[key]

    def get_embedding(self, key:int):
        return self.embeddings[key]

    def close(self):
        self.features.close()
        self.embeddings.close()
        self.infra1_images.close()
        self.depths.close()
        self.rgb_images.close()

class BagPlayer(Node):
    def __init__(self, bag_uri: str, storage_id: str = "sqlite3", serialization_format: str = "cdr",
                 realtime_rate: float = 0.0,
    ):
        super().__init__("rosbag_player")

        self._storage_options = StorageOptions(uri=bag_uri, storage_id="sqlite3",)
        self._converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr",)

        self._reader = SequentialReader()
        self._reader.open(self._storage_options, self._converter_options)

        self.start_timestamp_ns = None
        self.end_timestamp_ns = None

        # When realtime_rate > 0, play_next() paces publishing against wall
        # clock so perception_node has time to warm up and process frames. When
        # 0, behavior matches the legacy "drain as fast as possible" loop.
        self._realtime_rate = float(realtime_rate)
        self._play_wall_start: Optional[float] = None
        self._play_bag_start_ns: Optional[int] = None

        topic_infos = self._reader.get_all_topics_and_types()
        if len(topic_infos) == 0:
            raise ValueError(f"Bag {bag_uri} has no topics")

        self.start_timestamp_ns, self.end_timestamp_ns = self._scan_bag_time_range(
            bag_uri,
            storage_id,
            serialization_format,
        )

        # topic -> (publisher, msg_type)
        self._topic_publishers = {}

        # Build publishers for all topics in the bag
        for topic_info in topic_infos:
            msg_type = get_message(topic_info.type)
            pub = self.create_publisher(msg_type, topic_info.name, 10)
            self._topic_publishers[topic_info.name] = (pub, msg_type)

        self.get_logger().info("Bag topics and message types:")
        for topic_info in sorted(topic_infos, key=lambda t: t.name):
            self.get_logger().info(f"  {topic_info.name} -> {topic_info.type}")

        # /clock publisher (for use_sim_time)
        self._clock_pub = self.create_publisher(Clock, "/clock", 10)
        self._mapping_percent_pub = self.create_publisher(Float32, "/mapping/percent", 10)

        self.get_logger().info(f"BagPlayer opened bag: {bag_uri}")

    def _scan_bag_time_range(self, bag_uri: str, storage_id: str, serialization_format: str) -> tuple[int, int]:
        # We have not found a rosbag2_py API that exposes the bag time range directly,
        # so for now we scan the bag once to get the first and last message timestamps.
        scan_reader = SequentialReader()
        scan_reader.open(
            StorageOptions(uri=bag_uri, storage_id=storage_id),
            ConverterOptions(
                input_serialization_format=serialization_format,
                output_serialization_format=serialization_format,
            ),
        )

        first_timestamp_ns = None
        last_timestamp_ns = None
        while scan_reader.has_next():
            _, _, timestamp_ns = scan_reader.read_next()
            timestamp_ns = int(timestamp_ns)
            if first_timestamp_ns is None:
                first_timestamp_ns = timestamp_ns
            last_timestamp_ns = timestamp_ns

        if first_timestamp_ns is None or last_timestamp_ns is None:
            raise ValueError(f"Bag {bag_uri} has no messages")

        return first_timestamp_ns, last_timestamp_ns

    def _publish_percent(self, percent: float) -> None:
        msg = Float32()
        msg.data = float(percent)
        self._mapping_percent_pub.publish(msg)

    def _publish_percent_from_timestamp(self, timestamp_ns: int) -> None:
        percent = 100.0 * (timestamp_ns - self.start_timestamp_ns) / (self.end_timestamp_ns - self.start_timestamp_ns)
        self._publish_percent(percent)

    def play_next(self) -> bool:
        """
        Publish the next message from the bag.
        Returns False when there are no more messages.
        """
        if not self._reader.has_next():
            return False

        topic, serialized_msg, timestamp_ns = self._reader.read_next()
        self._publish_percent_from_timestamp(int(timestamp_ns))

        # Wall-clock pacing: hold messages back so the bag takes roughly its
        # recorded duration (divided by realtime_rate) to play out. This is
        # needed when a heavy downstream node (perception_node) must keep up.
        if self._realtime_rate > 0.0:
            if self._play_wall_start is None:
                self._play_wall_start = time.monotonic()
                self._play_bag_start_ns = int(timestamp_ns)
            else:
                bag_elapsed_s = (int(timestamp_ns) - self._play_bag_start_ns) / 1e9
                target_wall = self._play_wall_start + bag_elapsed_s / self._realtime_rate
                sleep_s = target_wall - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)

        # Find publisher + msg type for this topic
        pub_and_type = self._topic_publishers.get(topic)
        if pub_and_type is None:
            # No publisher (should not really happen, but don't crash playback)
            self.get_logger().warn(f"No publisher for topic '{topic}'")
            return True

        pub, msg_type = pub_and_type

        # Deserialize and publish actual message
        msg = deserialize_message(serialized_msg, msg_type)
        pub.publish(msg)

        # Publish /clock with the same timestamp (for use_sim_time)
        if self._clock_pub is not None:
            clock_msg = Clock()
            clock_msg.clock.sec = int(timestamp_ns // 1_000_000_000)
            clock_msg.clock.nanosec = int(timestamp_ns % 1_000_000_000)
            self._clock_pub.publish(clock_msg)

        return True

class BuildMapNode(Node):
    def __init__(
        self,
        map_save_path: str,
        verbose_timer: bool = True,
        has_rgb: bool = True,
        enable_ground_plane_constraint: bool = True,
    ):
        super().__init__('build_map_node')
        self.verbose_timer = verbose_timer
        self.has_rgb = has_rgb
        self.logger = logging.getLogger(__name__)
        self.timer_logger = self.logger.info if verbose_timer else self.logger.debug
        self.super_point_extractor = SuperPointTRT()
        self.light_glue_matcher = LightGlueTRT()
        self.dinov2_model = Dinov2TRT()

        self.bridge = CvBridge()

        self.tf_broadcaster = TransformBroadcaster(self)

        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/infra2/camera_info', self.info_callback, 10)
        self.depth_sub = Subscriber(self, Image, '/slam/keyframe_depth')
        self.keyframe_image_sub = Subscriber(self, Image, '/slam/keyframe_image')
        self.keyframe_odom_sub = Subscriber(self, Odometry, '/slam/keyframe_odom')
        self.continuous_odom_sub = self.create_subscription(Odometry, '/slam/odometry', self.continuous_odom_callback, 100)

        self.marker_pub = self.create_publisher(MarkerArray, '/mapping/pointcloud_markers', 10)
        self.local_map_pub = self.create_publisher(PointCloud2, "/mapping/local_map", 10)
        self.pose_graph_trajectory_pub = self.create_publisher(Path, "/mapping/pose_graph_trajectory", 10)
        self.project_3d_to_2d_pub = self.create_publisher(Image, "/mapping/project_3d_to_2d", 10)
        self.matches_image_pub = self.create_publisher(Image, "/mapping/keyframe_matches_images", 10)
        self.loop_matches_image_pub = self.create_publisher(Image, "/mapping/loop_matches_images", 10)
        self.global_map_marker_pub = self.create_publisher(MarkerArray, "/mapping/global_map_marker", 10)

        # Add stop signal subscription and save finished publisher
        self.mapping_stop_sub = self.create_subscription(Bool, '/benchmark/stop', self.mapping_stop_callback, 10)
        self.mapping_save_finished_pub = self.create_publisher(Bool, '/benchmark/data_saved', 10)
        # Keep sync queue bounded to reduce memory spikes/OOM risk on Jetson during map building.
        if self.has_rgb:
            self.rgb_image_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
            self.ts = ApproximateTimeSynchronizer([self.keyframe_image_sub, self.keyframe_odom_sub, self.depth_sub, self.rgb_image_sub], 200, 0.02)
            self.ts.registerCallback(self.keyframe_callback)
        else:
            self.rgb_image_sub = None
            self.ts = ApproximateTimeSynchronizer([self.keyframe_image_sub, self.keyframe_odom_sub, self.depth_sub], 200, 0.02)
            self.ts.registerCallback(self.keyframe_callback_no_rgb)

        self.K = None
        self.baseline = None
        self.odom = {}
        self.pose_graph_used_pose = {}
        self.relative_pose_constraint = []
        self.last_keyframe_timestamp = None
        self.enable_ground_plane_constraint = enable_ground_plane_constraint
        self.ground_planes = {}
        self.continuous_odom_recorder = OdomPoseRecorder(map_save_path, "mapping")

        os.makedirs(f"{map_save_path}", exist_ok=True)
        self.db = TinyNavDB(map_save_path)

        self.marker_id = 0

        self.loop_similarity_threshold = 0.90
        self.loop_top_k = 1

        self.map_save_path = map_save_path
        self._save_completed = False
        self.tf_sub = Subscriber(self, TFMessage, "/tf")
        self.tf_sub.registerCallback(self.tf_callback)
        self.tf_static_sub = Subscriber(self, TFMessage, "/tf_static")
        self.tf_static_sub.registerCallback(self.tf_callback)
        self.T_rgb_to_infra1 = None
        self.rgb_camera_K = None
        if self.has_rgb:
            self.rgb_camera_info_sub = Subscriber(self, CameraInfo, "/camera/camera/color/camera_info")
            self.rgb_camera_info_sub.registerCallback(self.rgb_camera_info_callback)
        else:
            self.rgb_camera_info_sub = None

    def tf_callback(self, msg:TFMessage):
        T_infra1_to_link = None
        T_infra1_optical_to_infra1 = None
        T_rgb_to_link = None
        T_rgb_optical_to_rgb = None
        tf_messages: Dict[int, Dict[str, np.ndarray]] = {}
        for t in msg.transforms:
            frame_id, child_frame_id, T = tf2np(t)
            timestamp_ns = int(t.header.stamp.sec * 1e9) + int(t.header.stamp.nanosec)
            if timestamp_ns not in tf_messages:
                tf_messages[timestamp_ns] = {}
            tf_messages[timestamp_ns][f"{frame_id}->{child_frame_id}"] = T
            if frame_id == "camera_link" and child_frame_id == "camera_infra1_frame":
                T_infra1_to_link = T
            if frame_id == "camera_infra1_frame" and child_frame_id == "camera_infra1_optical_frame":
                T_infra1_optical_to_infra1 = T
            if frame_id == "camera_color_frame" and child_frame_id == "camera_color_optical_frame":
                T_rgb_optical_to_rgb = T
            if frame_id == "camera_link" and child_frame_id == "camera_color_frame":
                T_rgb_to_link = T
            # Looper bags use cam_left/cam_rgb directly as camera frames.
            # In this code path, TF matrix is interpreted as child -> frame.
            if frame_id == "cam_left" and child_frame_id == "cam_rgb":
                self.T_rgb_to_infra1 = T

        if T_infra1_optical_to_infra1 is not None and T_rgb_optical_to_rgb is not None and T_infra1_to_link is not None and T_rgb_to_link is not None:
            self.T_rgb_to_infra1 = np.linalg.inv(T_infra1_optical_to_infra1) @ np.linalg.inv(T_infra1_to_link) @ T_rgb_to_link @ T_rgb_optical_to_rgb
        rgb_ready = (not self.has_rgb) or (self.T_rgb_to_infra1 is not None)
        if tf_messages and rgb_ready:
            np.save(f"{self.map_save_path}/tf_messages.npy", tf_messages, allow_pickle=True)
            if self.tf_sub is not None:
                self.destroy_subscription(self.tf_sub.sub)
                self.tf_sub = None
            if self.tf_static_sub is not None:
                self.destroy_subscription(self.tf_static_sub.sub)
                self.tf_static_sub = None
            self.get_logger().info("Saved tf_messages.npy and unsubscribed from /tf and /tf_static")

    def rgb_camera_info_callback(self, msg:CameraInfo):
        if self.rgb_camera_K is None:
            self.rgb_camera_K = np.array(msg.k).reshape(3, 3)

    def info_callback(self, msg:CameraInfo):
        if self.K is None:
            self.get_logger().info("Camera intrinsics received.")
            self.K = np.array(msg.k).reshape(3, 3)
            fx = self.K[0, 0]
            Tx = msg.p[3]
            self.baseline = -Tx / fx
            self.destroy_subscription(self.camera_info_sub)

    def continuous_odom_callback(self, odom_msg: Odometry):
        self.continuous_odom_recorder.record_odometry_msg(odom_msg)

    def mapping_stop_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Received benchmark stop signal, starting save process...")
            try:
                self.save_mapping()
                self.get_logger().info("Mapping save completed successfully")

                # Publish save finished signal
                save_finished_msg = Bool()
                save_finished_msg.data = True
                self.mapping_save_finished_pub.publish(save_finished_msg)
                self.get_logger().info("Published data save finished signal")

            except Exception as e:
                self.get_logger().error(f"Error during mapping save: {e}")
                # Still publish completion signal even if there was an error
                save_finished_msg = Bool()
                save_finished_msg.data = False
                self.mapping_save_finished_pub.publish(save_finished_msg)

    def keyframe_callback(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, depth_msg:Image, rgb_image_msg:Image):
        with Timer(name="Mapping Loop", text="[{name}] Elapsed time: {milliseconds:.0f} ms\n\n", logger=self.timer_logger):
            if self.K is None:
                return
            self.process(keyframe_image_msg, keyframe_odom_msg, depth_msg, rgb_image_msg)

    def keyframe_callback_no_rgb(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, depth_msg:Image):
        with Timer(name="Mapping Loop", text="[{name}] Elapsed time: {milliseconds:.0f} ms\n\n", logger=self.timer_logger):
            if self.K is None:
                return
            self.process(keyframe_image_msg, keyframe_odom_msg, depth_msg, None)

    def process(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, depth_msg:Image, rgb_image_msg):
        with Timer(name = "Msg decode", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            keyframe_image_timestamp = int(keyframe_image_msg.header.stamp.sec * 1e9) + int(keyframe_image_msg.header.stamp.nanosec)
            keyframe_odom_timestamp = int(keyframe_odom_msg.header.stamp.sec * 1e9) + int(keyframe_odom_msg.header.stamp.nanosec)
            keyframe_depth_timestamp = int(depth_msg.header.stamp.sec * 1e9) + int(depth_msg.header.stamp.nanosec)
            if keyframe_image_timestamp != keyframe_odom_timestamp or keyframe_image_timestamp != keyframe_depth_timestamp:
                self.get_logger().error(f"Keyframe timestamp mismatch: {keyframe_image_timestamp} != {keyframe_odom_timestamp} != {keyframe_depth_timestamp}")

            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            odom, _ = msg2np(keyframe_odom_msg)
            infra1_image = self.bridge.imgmsg_to_cv2(keyframe_image_msg, desired_encoding="mono8")
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, desired_encoding="bgr8") if rgb_image_msg is not None else None

        with Timer(name = "save image and depth", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            self.db.set_entry(keyframe_image_timestamp, depth = depth, infra1_image = infra1_image, rgb_image = rgb_image)

        if self.enable_ground_plane_constraint:
            with Timer(name = "ground plane fit", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                ground_plane = estimate_ground_plane_from_depth(depth, self.K)
                if ground_plane is not None:
                    self.ground_planes[keyframe_image_timestamp] = ground_plane

        with Timer(name = "get embeddings", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            embedding = self.get_embeddings(infra1_image)
            embedding = embedding / np.linalg.norm(embedding)
            self.db.set_entry(keyframe_image_timestamp, embedding = embedding)
        with Timer(name = "super point extractor", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            features = asyncio.run(self.super_point_extractor.infer(infra1_image))
            self.db.set_entry(keyframe_image_timestamp, features = features)

        with Timer(name = "loop and pose graph solve", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            if len(self.odom) == 0 and self.last_keyframe_timestamp is None:
                self.odom[keyframe_image_timestamp] = odom
                self.pose_graph_used_pose[keyframe_image_timestamp] = odom
            else:
                last_keyframe_odom_pose = self.odom[self.last_keyframe_timestamp]
                T_prev_curr = np.linalg.inv(last_keyframe_odom_pose) @ odom
                self.relative_pose_constraint.append((keyframe_image_timestamp, self.last_keyframe_timestamp, T_prev_curr))
                self.pose_graph_used_pose[keyframe_image_timestamp] = odom
                self.odom[keyframe_image_timestamp] = odom


                def find_loop_and_pose_graph(timestamp):
                    target_embedding = self.db.get_embedding(timestamp)
                    valid_timestamp = [t for t in self.pose_graph_used_pose.keys() if t + 10 * 1e9 < timestamp]
                    valid_embeddings = np.array([self.db.get_embedding(t) for t in valid_timestamp])

                    idx_to_timestamp = {i:t for i, t in enumerate(valid_timestamp)}
                    with Timer(name = "find loop", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        loop_list = find_loop(target_embedding, valid_embeddings, self.loop_similarity_threshold, self.loop_top_k)
                    with Timer(name = "Relative pose estimation", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        for idx, similarity in loop_list:
                            prev_timestamp = idx_to_timestamp[idx]
                            curr_timestamp = timestamp
                            prev_depth, _, prev_features, _, _ = self.db.get_depth_embedding_features_images(prev_timestamp)
                            curr_depth, _, curr_features, _, _ = self.db.get_depth_embedding_features_images(curr_timestamp)
                            prev_matched_keypoints, curr_matched_keypoints, matches = self.match_keypoints(prev_features, curr_features)
                            success, T_prev_curr, _, _, inliers = estimate_pose(prev_matched_keypoints, curr_matched_keypoints, curr_depth, self.K)
                            if success and len(inliers) >= 100:
                                self.relative_pose_constraint.append((curr_timestamp, prev_timestamp, T_prev_curr))
                                print(f"Added loop relative pose constraint: {curr_timestamp} -> {prev_timestamp}")
                    with Timer(name = "solve pose graph", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        self.pose_graph_used_pose = solve_pose_graph(self.pose_graph_used_pose, self.relative_pose_constraint, max_iteration_num = 5)
                find_loop_and_pose_graph(keyframe_image_timestamp)

        with Timer(name = "publish local pointcloud", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            cloud = depth_to_cloud(depth, self.K, 30, 3)
            self.publish_local_map(cloud, 'camera_'+str(keyframe_image_timestamp))

        with Timer(name = "tf publish", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            self.publish_all_transforms()

        with Timer(name = "pose graph trajectory publish", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            self.pose_graph_trajectory_publish(keyframe_image_timestamp)
        self.last_keyframe_timestamp = keyframe_image_timestamp

    def get_embeddings(self, image: np.ndarray) -> np.ndarray:
        # shape: (1, 768)
        return asyncio.run(self.dinov2_model.infer(image))

    def match_keypoints(self, feats0:dict, feats1:dict, image_shape = np.array([848, 480], dtype = np.int64)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        match_result = asyncio.run(self.light_glue_matcher.infer(feats0["kpts"], feats1["kpts"], feats0['descps'], feats1['descps'], feats0['mask'], feats1['mask'], image_shape, image_shape))
        match_indices = match_result["match_indices"][0]
        valid_mask = match_indices != -1
        keypoints0 = feats0["kpts"][0][valid_mask]
        keypoints1 = feats1["kpts"][0][match_indices[valid_mask]]
        matches = []
        for i, index in enumerate(match_indices):
            if index != -1:
                matches.append([i, index])
        return keypoints0, keypoints1, np.array(matches, dtype=np.int64)

    def pose_graph_trajectory_publish(self, timestamp):
        path_msg = Path()
        path_msg.header.stamp.sec = int(timestamp / 1e9)
        path_msg.header.stamp.nanosec = int(timestamp % 1e9)
        path_msg.header.frame_id = "world"
        for t, pose_in_world in self.pose_graph_used_pose.items():
            pose = PoseStamped()
            pose.header = path_msg.header
            t = pose_in_world[:3, 3]
            quat = matrix_to_quat(pose_in_world[:3, :3])
            pose.pose.position.x = t[0]
            pose.pose.position.y = t[1]
            pose.pose.position.z = t[2]
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            path_msg.poses.append(pose)
        self.pose_graph_trajectory_pub.publish(path_msg)

    def save_mapping(self):
        if self._save_completed:
            self.get_logger().info("Mapping data already saved, skipping duplicate save")
            return

        if self.K is None:
            self.get_logger().info("No camera intrinsics available, skipping save")
            return

        self.get_logger().info("Saving mapping data...")

        # Save continuous poses
        self.continuous_odom_recorder.save_to_disk()

        with Timer(name = "final pose graph", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            self.pose_graph_used_pose = solve_pose_graph(self.pose_graph_used_pose, self.relative_pose_constraint)

        if self.enable_ground_plane_constraint:
            with Timer(name = "ground plane constraint", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                self.pose_graph_used_pose, ground_stats = apply_ground_plane_constraint(
                    self.pose_graph_used_pose,
                    self.ground_planes,
                )
                np.save(f"{self.map_save_path}/ground_plane_stats.npy", ground_stats, allow_pickle=True)
                if ground_stats["enabled"]:
                    self.get_logger().info(
                        "Ground plane constraint: "
                        f"corrected={ground_stats['corrected_count']}, "
                        f"skipped_no_plane={ground_stats['skipped_no_plane']}, "
                        f"skipped_tilt={ground_stats['skipped_tilt']}, "
                        f"skipped_height={ground_stats['skipped_height']}"
                    )
                else:
                    self.get_logger().warning("Ground plane constraint disabled: not enough reliable ground planes")

        np.save(f"{self.map_save_path}/poses.npy", self.pose_graph_used_pose, allow_pickle = True)
        np.save(f"{self.map_save_path}/intrinsics.npy", self.K)
        np.save(f"{self.map_save_path}/baseline.npy", self.baseline)
        if self.has_rgb and self.T_rgb_to_infra1 is not None:
            print(f"T_rgb_to_infra1: {self.T_rgb_to_infra1}")
            np.save(f"{self.map_save_path}/T_rgb_to_infra1.npy", self.T_rgb_to_infra1, allow_pickle = True)
        if self.has_rgb and self.rgb_camera_K is not None:
            np.save(f"{self.map_save_path}/rgb_camera_intrinsics.npy", self.rgb_camera_K, allow_pickle = True)

        # Flush and close writable DB first, then reopen DB for occupancy generation.
        self.db.close()
        occupancy_db = TinyNavDB(self.map_save_path, is_scratch=False)

        # Generate occupancy map
        occupancy_resolution = 0.1
        occupancy_step = 10
        occupancy_grid, occupancy_origin, occupancy_2d_image, sdf_map = generate_occupancy_map(
            self.pose_graph_used_pose, occupancy_db, self.K, self.baseline, occupancy_resolution, occupancy_step
        )
        occupancy_db.close()
        occupancy_meta = np.array([occupancy_origin[0], occupancy_origin[1], occupancy_origin[2], occupancy_resolution], dtype=np.float32)
        np.save(f"{self.map_save_path}/occupancy_grid.npy", occupancy_grid)
        np.save(f"{self.map_save_path}/occupancy_meta.npy", occupancy_meta)
        np.save(f"{self.map_save_path}/sdf_map.npy", sdf_map)
        cv2.imwrite(f"{self.map_save_path}/occupancy_2d_image.png", occupancy_2d_image)

        self._save_completed = True
        self.get_logger().info("Full mapping data saved successfully")


    def pointcloud_to_marker_array(self, points, frame_id='camera',colors=None):
        marker_array = MarkerArray()
        
        # Create point cloud Marker
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pointcloud"
        marker.id = self.marker_id
        self.marker_id = self.marker_id + 1

        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # Set Marker properties
        marker.scale.x = 0.03  # Point width
        marker.scale.y = 0.03  # Point height
        marker.scale.z = 0.0   # For POINTS type, z is not used
        
        # Set orientation (unit quaternion)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Set position
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        
        # Set points
        marker.points = []
        for point in points:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = float(point[2])
            if (p.y > 0):
                marker.points.append(p)
                c = z_value_to_color(float(point[1]), -3, 1)
                marker.colors.append(c)
        
        # Set lifetime (0 means never expire)
        marker.lifetime.sec = 0
        marker.frame_locked = True
        
        marker_array.markers.append(marker) 
        
        return marker_array

    def publish_local_map(self, point_cloud, frame_id):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        marker_array = self.pointcloud_to_marker_array(point_cloud.tolist(), frame_id)
        self.marker_pub.publish(marker_array)

    def publish_all_transforms(self):
        """Publish all pose TF transforms"""
        if not self.pose_graph_used_pose:
            return
            
        transforms = []        
        for time, pose_in_world in self.pose_graph_used_pose.items():
            transform = TransformStamped()
            
            # Set header
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = 'world'
            transform.child_frame_id = 'camera_' + str(time)
            
            # Set position
            t = pose_in_world[:3, 3]
            transform.transform.translation.x = t[0]
            transform.transform.translation.y = t[1]
            transform.transform.translation.z = t[2]
            qx,qy,qz,qw =  R.from_matrix(pose_in_world[:3, :3]).as_quat()
            transform.transform.rotation.x = qx
            transform.transform.rotation.y = qy
            transform.transform.rotation.z = qz
            transform.transform.rotation.w = qw

            transforms.append(transform)
        
        # Publish all TF transforms
        self.tf_broadcaster.sendTransform(transforms)
        

    def destroy_node(self):
        try:
            self.save_mapping()
            super().destroy_node()
        except Exception:
            # Ignore errors during destruction as resources may already be freed
            pass

class ImageTransportsNode(Node):
    def __init__(self):
        super().__init__('image_transports_node')
        # Simple compressed → raw image transport for color images.
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_rect_raw/compressed',
            self.image_callback,
            10,
        )
        self.image_pub = self.create_publisher(Image, '/camera/camera/color/image_raw', 10)
        self.bridge = CvBridge()

    def image_callback(self, msg: CompressedImage):
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        image_msg.header.stamp = msg.header.stamp
        image_msg.header.frame_id = msg.header.frame_id
        self.image_pub.publish(image_msg)

def main(args=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    rclpy.init(args=args)


    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_file", type=str, default="tinynav_db")
    parser.add_argument("--map_save_path", type=str, default="tinynav_db")
    parser.add_argument("--verbose_timer", action="store_true", default=True, help="Enable verbose timer output")
    parser.add_argument("--no_verbose_timer", dest="verbose_timer", action="store_false", help="Disable verbose timer output")
    parser.add_argument("--realtime_rate", type=float, default=0.0,
                        help="If >0, pace BagPlayer against wall clock at this rate (1.0 = real-time). "
                             "Default 0 keeps legacy max-throughput playback.")
    parser.add_argument(
        "--ground_plane_constraint",
        action="store_true",
        default=True,
        help="Constrain final map poses with per-keyframe ground plane observations.",
    )
    parser.add_argument(
        "--no_ground_plane_constraint",
        dest="ground_plane_constraint",
        action="store_false",
        help="Disable ground plane pose correction during map save.",
    )
    parsed_args, unknown_args = parser.parse_known_args(sys.argv[1:])

    exec_ = SingleThreadedExecutor()
    player_node = BagPlayer(parsed_args.bag_file, realtime_rate=parsed_args.realtime_rate)
    bag_topics = set(player_node._topic_publishers.keys())
    has_rgb = '/camera/camera/color/image_raw' in bag_topics or '/camera/camera/color/image_rect_raw/compressed' in bag_topics
    has_compressed_rgb = '/camera/camera/color/image_rect_raw/compressed' in bag_topics
    if not has_rgb:
        player_node.get_logger().info("Bag has no RGB topics; building grayscale-only map.")
    map_node = BuildMapNode(
        parsed_args.map_save_path,
        verbose_timer=parsed_args.verbose_timer,
        has_rgb=has_rgb,
        enable_ground_plane_constraint=parsed_args.ground_plane_constraint,
    )
    exec_.add_node(player_node)
    exec_.add_node(map_node)
    if has_compressed_rgb:
        image_transports_node = ImageTransportsNode()
        exec_.add_node(image_transports_node)
    while rclpy.ok() and player_node.play_next():
        exec_.spin_once(timeout_sec=0.001)
    player_node._publish_percent(100.0)
    map_node.save_mapping()

if __name__ == '__main__':
    main()
