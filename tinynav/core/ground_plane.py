from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def rotation_from_two_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = normalize_vector(src.astype(np.float64))
    dst = normalize_vector(dst.astype(np.float64))
    cross = np.cross(src, dst)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if cross_norm < 1e-8:
        if dot > 0.0:
            return np.eye(3)
        axis = np.cross(src, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0]))
        axis = normalize_vector(axis)
        return R.from_rotvec(np.pi * axis).as_matrix()
    axis = cross / cross_norm
    angle = np.arctan2(cross_norm, dot)
    return R.from_rotvec(angle * axis).as_matrix()


def _fit_plane_pca(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize_vector(vh[-1])
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


def transform_plane_to_world(plane: dict, pose: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    normal = pose[:3, :3] @ plane["normal"].astype(np.float64)
    normal = normalize_vector(normal)
    centroid = pose[:3, :3] @ plane["centroid"].astype(np.float64) + pose[:3, 3]
    offset = float(plane["offset"] - np.dot(normal, pose[:3, 3]))
    return normal, offset, centroid


def estimate_global_ground_plane(poses: dict, ground_planes: dict) -> Optional[dict]:
    centroids = []
    normals = []
    for timestamp, plane in ground_planes.items():
        if timestamp not in poses:
            continue
        normal, _, centroid = transform_plane_to_world(plane, poses[timestamp])
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
    mean_normal = normalize_vector(np.mean(normals, axis=0))
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
            mean_normal = normalize_vector(normal + mean_normal)

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
        "source_count": len(centroids),
    }
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

    global_normal = normalize_vector(global_plane["normal"].astype(np.float64))
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
        normal_world = normalize_vector(pose[:3, :3] @ normal_cam)
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
        correction_rot = rotation_from_two_vectors(normal_world, global_normal)
        corrected_pose[:3, :3] = correction_rot @ pose[:3, :3]
        corrected_pose[:3, 3] = pose[:3, 3] + height_delta * global_normal
        corrected_poses[timestamp] = corrected_pose
        stats["corrected_count"] += 1
        stats["tilt_correction_deg"].append(float(np.rad2deg(tilt)))
        stats["height_correction"].append(height_delta)

    stats["tilt_correction_deg"] = np.asarray(stats["tilt_correction_deg"], dtype=np.float32)
    stats["height_correction"] = np.asarray(stats["height_correction"], dtype=np.float32)
    return corrected_poses, stats


class HorizontalGroundPlaneTracker:
    def __init__(
        self,
        world_normal: np.ndarray | None = None,
        offset_alpha: float = 0.05,
        max_tilt_correction_deg: float = 10.0,
        max_height_correction: float = 0.50,
    ):
        self.world_normal = normalize_vector(
            np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if world_normal is None
            else world_normal.astype(np.float64)
        )
        self.offset_alpha = float(offset_alpha)
        self.max_tilt_rad = float(np.deg2rad(max_tilt_correction_deg))
        self.max_height_correction = float(max_height_correction)
        self.reference_offset: float | None = None

    def observe(self, plane: Optional[dict], pose: np.ndarray) -> dict:
        _, stats = self.make_prior_pose(plane, pose, update_reference=True)
        return stats

    def make_prior_pose(
        self,
        plane: Optional[dict],
        pose: np.ndarray,
        update_reference: bool = False,
    ) -> tuple[Optional[np.ndarray], dict]:
        stats = {
            "enabled": False,
            "reason": "no_plane",
            "tilt_deg": 0.0,
            "height_delta": 0.0,
            "reference_offset": self.reference_offset,
        }
        if plane is None:
            return None, stats

        normal_cam = plane["normal"].astype(np.float64)
        offset_cam = float(plane["offset"])
        normal_world = normalize_vector(pose[:3, :3] @ normal_cam)
        if np.dot(normal_world, self.world_normal) < 0.0:
            normal_cam = -normal_cam
            offset_cam = -offset_cam
            normal_world = -normal_world

        tilt = float(np.arccos(np.clip(np.dot(normal_world, self.world_normal), -1.0, 1.0)))
        stats["tilt_deg"] = float(np.rad2deg(tilt))
        if tilt > self.max_tilt_rad:
            stats["reason"] = "tilt"
            return None, stats

        measured_offset = float(offset_cam - np.dot(self.world_normal, pose[:3, 3]))
        reference_offset = self.reference_offset
        if reference_offset is None:
            if not update_reference:
                stats["reason"] = "no_reference"
                return None, stats
            reference_offset = measured_offset
            self.reference_offset = measured_offset

        height_delta = float(offset_cam - reference_offset - np.dot(self.world_normal, pose[:3, 3]))
        stats["height_delta"] = height_delta
        stats["reference_offset"] = reference_offset
        if abs(height_delta) > self.max_height_correction:
            stats["reason"] = "height"
            return None, stats

        if update_reference:
            self.reference_offset = (
                (1.0 - self.offset_alpha) * self.reference_offset
                + self.offset_alpha * measured_offset
            )
            stats["reference_offset"] = self.reference_offset

        prior_pose = pose.copy()
        correction_rot = rotation_from_two_vectors(normal_world, self.world_normal)
        prior_pose[:3, :3] = correction_rot @ pose[:3, :3]
        prior_pose[:3, 3] = pose[:3, 3] + height_delta * self.world_normal
        stats["enabled"] = True
        stats["reason"] = "ok"
        return prior_pose, stats
