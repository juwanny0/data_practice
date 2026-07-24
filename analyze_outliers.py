"""LeRobot 에피소드의 범용 데이터 품질 및 이상 수행 후보 분석.

이 모듈은 이상한 움직임을 곧바로 실패로 단정하지 않는다. 데이터 자체가
깨졌거나 여러 독립적인 품질 신호가 동시에 발생한 경우만 ``delete``로,
희귀하지만 정상일 수 있는 수행은 ``validate``로 분류한다.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from execution_logging import start_automatic_logging

if __name__ == "__main__":
    start_automatic_logging()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROGRAM_STARTED_AT = time.perf_counter()


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BASE_DIR / "Task_000200_Put_PotLib_Lift13_CBG_lerobot_fail_example"
DEFAULT_TASK_FEATURE_PREFIXES = ("arm_", "gripper_")
MOTION_BLOCK_NAMES = (
    "median_abs_velocity",
    "q95_abs_velocity",
    "median_abs_acceleration",
    "q95_abs_acceleration",
    "position_iqr",
    "active_fraction",
    "end_displacement",
)


def resample_trajectory(trajectory: np.ndarray, target_length: int = 60) -> np.ndarray:
    """길이가 다른 궤적을 고정 길이로 선형 리샘플링한다."""
    old_time = np.linspace(0.0, 1.0, len(trajectory))
    new_time = np.linspace(0.0, 1.0, target_length)
    return np.column_stack([
        np.interp(new_time, old_time, trajectory[:, feature_idx])
        for feature_idx in range(trajectory.shape[1])
    ])


def multivariate_dtw_distance(
    first: np.ndarray,
    second: np.ndarray,
    window: int = 12,
) -> float:
    """Sakoe-Chiba window를 적용한 다변량 DTW 평균 비용을 계산한다."""
    local_cost = cdist(first, second, metric="euclidean")
    rows, cols = local_cost.shape
    window = max(window, abs(rows - cols))
    accumulated = np.full((rows + 1, cols + 1), np.inf)
    accumulated[0, 0] = 0.0
    for row in range(1, rows + 1):
        start = max(1, row - window)
        end = min(cols, row + window)
        for col in range(start, end + 1):
            accumulated[row, col] = local_cost[row - 1, col - 1] + min(
                accumulated[row - 1, col],
                accumulated[row, col - 1],
                accumulated[row - 1, col - 1],
            )
    return float(accumulated[rows, cols] / (rows + cols))


def _robust_upper(values: np.ndarray, multiplier: float = 1.5) -> float:
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return float(q3 + multiplier * (q3 - q1))


def _robust_lower(values: np.ndarray, multiplier: float = 1.5) -> float:
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return float(q1 - multiplier * (q3 - q1))


def _flat_state_columns(columns: Sequence[str]) -> list[str]:
    candidates = [column for column in columns if column.startswith("observation.state")]

    def sort_key(column: str) -> tuple[int, str]:
        suffix = column.rsplit("_", 1)[-1]
        return (int(suffix), column) if suffix.isdigit() else (10**9, column)

    return sorted(candidates, key=sort_key)


def load_dataset_info(dataset_dir: Path) -> tuple[dict[str, Any], list[str]]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"LeRobot 메타데이터를 찾을 수 없습니다: {info_path}")
    with info_path.open(encoding="utf-8") as info_file:
        info = json.load(info_file)
    state_info = info.get("features", {}).get("observation.state")
    if not state_info:
        raise ValueError("meta/info.json에 observation.state 정의가 없습니다.")
    state_names = state_info.get("names")
    state_shape = state_info.get("shape")
    if not state_names or not state_shape or len(state_names) != int(state_shape[0]):
        raise ValueError("observation.state의 shape와 names가 유효하지 않습니다.")
    return info, list(state_names)


def discover_episode_files(dataset_dir: Path) -> list[Path]:
    files = sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"에피소드 Parquet을 찾을 수 없습니다: {dataset_dir}/data/chunk-*/episode_*.parquet"
        )
    return files


def _episode_id_from_path(filepath: Path) -> int:
    return int(filepath.stem.rsplit("_", 1)[-1])


def _motion_blocks(joint_data: np.ndarray) -> np.ndarray:
    """순간 최대값 대신 중앙값과 95% 분위수를 사용한 강건한 운동 특징."""
    width = joint_data.shape[1]
    velocity = np.abs(np.diff(joint_data, axis=0))
    acceleration = np.abs(np.diff(joint_data, n=2, axis=0))
    if not len(velocity):
        velocity = np.zeros((1, width))
    if not len(acceleration):
        acceleration = np.zeros((1, width))
    q95_velocity = np.quantile(velocity, 0.95, axis=0)
    activity_floor = np.maximum(q95_velocity * 0.10, 1e-6)
    active_fraction = np.mean(velocity > activity_floor, axis=0)
    return np.vstack([
        np.median(velocity, axis=0),
        q95_velocity,
        np.median(acceleration, axis=0),
        np.quantile(acceleration, 0.95, axis=0),
        np.quantile(joint_data, 0.75, axis=0) - np.quantile(joint_data, 0.25, axis=0),
        active_fraction,
        np.abs(joint_data[-1] - joint_data[0]),
    ])


def extract_episode_features(
    parquet_files: Sequence[Path],
    state_names: Sequence[str],
    task_feature_indices: np.ndarray,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], list[str], dict[int, str]]:
    summaries: list[dict[str, Any]] = []
    trajectories: dict[int, np.ndarray] = {}
    warnings: list[str] = []
    integrity_failures: dict[int, str] = {}
    expected_width = len(state_names)

    for filepath in parquet_files:
        fallback_idx = _episode_id_from_path(filepath)
        try:
            frame = pd.read_parquet(filepath)
            if frame.empty:
                raise ValueError("빈 에피소드")
            episode_idx = (
                int(frame["episode_index"].iloc[0])
                if "episode_index" in frame.columns else fallback_idx
            )
            if episode_idx in trajectories:
                raise ValueError(f"중복 episode_index: {episode_idx}")
            if "observation.state" in frame.columns:
                joint_data = np.asarray(frame["observation.state"].tolist(), dtype=float)
            else:
                state_columns = _flat_state_columns(frame.columns)
                if not state_columns:
                    raise ValueError("observation.state 컬럼이 없음")
                joint_data = frame[state_columns].to_numpy(dtype=float)
            if joint_data.ndim != 2 or joint_data.shape[1] != expected_width:
                raise ValueError(
                    f"observation.state shape 불일치: {joint_data.shape}, expected (*, {expected_width})"
                )
            if not np.isfinite(joint_data).all():
                raise ValueError("observation.state에 NaN/Inf가 있음")

            blocks = _motion_blocks(joint_data)
            task_blocks = blocks[:, task_feature_indices]
            trajectories[episode_idx] = joint_data
            summaries.append({
                "episode_idx": episode_idx,
                "length": len(frame),
                "min_joints": np.min(joint_data, axis=0),
                "max_joints": np.max(joint_data, axis=0),
                "motion_features": task_blocks.reshape(-1),
                "full_motion_features": blocks.reshape(-1),
            })
        except Exception as error:
            message = str(error)
            integrity_failures[fallback_idx] = message
            warnings.append(f"{filepath}: {message}")

    if not summaries:
        raise RuntimeError("분석 가능한 에피소드가 없습니다. 경로와 스키마를 확인하세요.")
    summary = pd.DataFrame(summaries).sort_values("episode_idx").reset_index(drop=True)
    if verbose:
        for warning in warnings:
            print(f"[경고] {warning}")
    return summary, trajectories, warnings, integrity_failures


def _annotation_profile(dataset_dir: Path, episode_ids: Sequence[int]) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for episode_idx in episode_ids:
        matches = list(dataset_dir.glob(f"annotations/chunk-*/episode_{episode_idx:06d}.json"))
        row = {
            "episode_idx": episode_idx,
            "annotation_available": False,
            "subtask_count": np.nan,
            "skill_count": np.nan,
            "annotation_coverage": np.nan,
        }
        if matches:
            try:
                payload = json.loads(matches[0].read_text(encoding="utf-8"))
                subtasks = payload.get("sub_task_annotation", [])
                skills = payload.get("skill_annotation", [])
                duration = float(payload.get("meta_data", {}).get("task_duration", 0))
                covered = 0.0
                for subtask in subtasks:
                    bounds = subtask.get("frame_duration", [])
                    if len(bounds) == 2:
                        covered += max(0.0, float(bounds[1]) - float(bounds[0]))
                row.update({
                    "annotation_available": True,
                    "subtask_count": len(subtasks),
                    "skill_count": len(skills),
                    "annotation_coverage": covered / duration if duration > 0 else 0.0,
                })
            except Exception as error:
                warnings.append(f"annotation episode {episode_idx}: {error}")
        rows.append(row)
    return pd.DataFrame(rows), warnings


def _hand_groups(joint_names: Sequence[str], task_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    task_names = [joint_names[idx] for idx in task_indices]
    left = np.array([i for i, name in enumerate(task_names) if "_l_" in name], dtype=int)
    right = np.array([i for i, name in enumerate(task_names) if "_r_" in name], dtype=int)
    return left, right


def _assign_hand_modes(
    feature_matrix: np.ndarray,
    task_joint_count: int,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    blocks = feature_matrix.reshape(len(feature_matrix), len(MOTION_BLOCK_NAMES), task_joint_count)
    # 중앙 속도는 정지 프레임이 많은 조작에서 0이 되므로 q95와 position IQR을 함께 사용한다.
    activity = blocks[:, 1, :] + blocks[:, 4, :]
    if not len(left_indices) or not len(right_indices):
        return np.full(len(feature_matrix), "all", dtype=object), np.full(len(feature_matrix), 0.5)
    left_activity = activity[:, left_indices].sum(axis=1)
    right_activity = activity[:, right_indices].sum(axis=1)
    share = left_activity / np.maximum(left_activity + right_activity, 1e-12)
    modes = np.where(share < 0.35, "right_dominant", np.where(
        share > 0.65, "left_dominant", "balanced"
    ))
    return modes.astype(object), share


def _temporal_activity_features(
    trajectories: Sequence[np.ndarray],
    joint_names: Sequence[str],
    task_indices: np.ndarray,
    bin_count: int = 4,
) -> np.ndarray:
    """태스크에 종속적인 단계명 없이 동작 순서를 요약하는 시간 구간 특징."""
    task_names = [joint_names[idx] for idx in task_indices]
    groups = [
        np.array([i for i, name in enumerate(task_names) if "arm_l_" in name], dtype=int),
        np.array([i for i, name in enumerate(task_names) if "arm_r_" in name], dtype=int),
        np.array([i for i, name in enumerate(task_names) if "gripper_" in name], dtype=int),
    ]
    rows: list[np.ndarray] = []
    for trajectory in trajectories:
        velocity = np.abs(np.diff(trajectory[:, task_indices], axis=0))
        boundaries = np.linspace(0, len(velocity), bin_count + 1, dtype=int)
        profile = np.zeros((len(groups), bin_count), dtype=float)
        for group_idx, indices in enumerate(groups):
            if not len(indices):
                continue
            for bin_idx in range(bin_count):
                segment = velocity[boundaries[bin_idx]:boundaries[bin_idx + 1], indices]
                profile[group_idx, bin_idx] = float(np.sum(segment))
            total = profile[group_idx].sum()
            if total > 1e-12:
                profile[group_idx] /= total
        rows.append(profile.reshape(-1))
    return np.vstack(rows)


def _adaptive_pca(
    feature_matrix: np.ndarray,
    variance_target: float = 0.90,
    max_components: int = 24,
) -> tuple[np.ndarray, StandardScaler, PCA | None, np.ndarray]:
    """상수 feature를 제거하고 설명분산 목표에 따라 PCA 차원을 자동 선택한다."""
    variances = np.var(feature_matrix, axis=0)
    keep = variances > 1e-12
    if not np.any(keep):
        keep[:] = True
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix[:, keep])
    max_rank = min(max_components, len(scaled) - 1, scaled.shape[1])
    if max_rank < 2:
        return scaled, scaler, None, keep
    probe = PCA(n_components=max_rank, random_state=42).fit(scaled)
    cumulative = np.cumsum(probe.explained_variance_ratio_)
    component_count = min(max_rank, max(2, int(np.searchsorted(cumulative, variance_target) + 1)))
    pca = PCA(n_components=component_count, random_state=42)
    return pca.fit_transform(scaled), scaler, pca, keep


def _mode_aware_isolation(
    feature_matrix: np.ndarray,
    modes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    scores = np.full(len(feature_matrix), np.nan)
    labels = np.ones(len(feature_matrix), dtype=int)
    thresholds: dict[str, float] = {}
    for mode in sorted(set(modes)):
        rows = np.flatnonzero(modes == mode)
        # 작은 정상 모드를 희귀하다는 이유만으로 이상치 처리하지 않는다.
        if len(rows) < 12:
            thresholds[str(mode)] = float("nan")
            continue
        scaled = StandardScaler().fit_transform(feature_matrix[rows])
        model = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=42,
        ).fit(scaled)
        mode_scores = model.score_samples(scaled)
        threshold = _robust_lower(mode_scores, 1.5)
        scores[rows] = mode_scores
        labels[rows] = np.where(mode_scores < threshold, -1, 1)
        thresholds[str(mode)] = threshold
    return scores, labels, thresholds


def _mode_aware_hdbscan(
    feature_matrix: np.ndarray,
    modes: np.ndarray,
    variance_target: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    global_embedding, _, global_pca, _ = _adaptive_pca(feature_matrix, variance_target)
    labels = np.full(len(feature_matrix), -2, dtype=int)
    probabilities = np.ones(len(feature_matrix), dtype=float)
    confirmed_noise = np.zeros(len(feature_matrix), dtype=bool)
    diagnostics: dict[str, Any] = {}
    label_offset = 0
    for mode in sorted(set(modes)):
        rows = np.flatnonzero(modes == mode)
        minimum_group = 8
        if len(rows) < minimum_group:
            labels[rows] = label_offset
            diagnostics[str(mode)] = {
                "size": len(rows), "status": "protected_small_mode", "outliers": []
            }
            label_offset += 1
            continue
        embedding, _, pca, _ = _adaptive_pca(feature_matrix[rows], variance_target)
        min_cluster_size = max(5, int(np.ceil(np.sqrt(len(rows)))))
        min_samples = max(3, min_cluster_size // 2)
        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method="eom",
            allow_single_cluster=True,
            copy=True,
        ).fit(embedding)
        local_labels = model.labels_.copy()
        positive = local_labels >= 0
        labels[rows[positive]] = local_labels[positive] + label_offset
        labels[rows[~positive]] = -1
        probabilities[rows] = model.probabilities_
        neighbors = NearestNeighbors(n_neighbors=min_samples).fit(embedding)
        local_density_distance = neighbors.kneighbors(embedding)[0][:, -1]
        density_threshold = _robust_upper(local_density_distance, 1.5)
        strong_local_noise = (local_labels == -1) & (
            local_density_distance > density_threshold
        )
        confirmed_noise[rows] = strong_local_noise
        cluster_count = len(set(local_labels[positive]))
        label_offset += max(1, cluster_count)
        diagnostics[str(mode)] = {
            "size": len(rows),
            "pca_components": None if pca is None else int(pca.n_components_),
            "explained_variance": None if pca is None else float(np.sum(pca.explained_variance_ratio_)),
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_count": cluster_count,
            "outlier_count": int(np.sum(local_labels == -1)),
            "confirmed_outlier_count": int(np.sum(strong_local_noise)),
            "local_density_threshold": density_threshold,
        }
    diagnostics["global_pca_components"] = None if global_pca is None else int(global_pca.n_components_)
    diagnostics["global_explained_variance"] = (
        None if global_pca is None else float(np.sum(global_pca.explained_variance_ratio_))
    )
    return labels, probabilities, confirmed_noise, global_embedding, diagnostics


def _mode_aware_dtw(
    trajectories: list[np.ndarray],
    modes: np.ndarray,
    target_length: int,
    window: int,
    verbose: bool,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict[str, float]]:
    relative = [trajectory - trajectory[0] for trajectory in trajectories]
    frames = np.vstack(relative)
    center = np.median(frames, axis=0)
    scale = np.quantile(frames, 0.75, axis=0) - np.quantile(frames, 0.25, axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = [
        resample_trajectory((trajectory - center) / scale, target_length)
        for trajectory in relative
    ]
    scores = np.full(len(normalized), np.nan)
    outlier_mask = np.zeros(len(normalized), dtype=bool)
    thresholds: dict[str, float] = {}
    if verbose:
        print("손 사용 모드별 DTW 궤적 거리 계산 중...")
    for mode in sorted(set(modes)):
        rows = np.flatnonzero(modes == mode)
        if len(rows) < 6:
            thresholds[str(mode)] = float("nan")
            continue
        distances = np.zeros((len(rows), len(rows)), dtype=np.float32)
        for first in range(len(rows)):
            for second in range(first + 1, len(rows)):
                distance = multivariate_dtw_distance(
                    normalized[rows[first]], normalized[rows[second]], window
                )
                distances[first, second] = distance
                distances[second, first] = distance
        neighbor_count = min(5, len(rows) - 1)
        mode_scores = np.sort(distances, axis=1)[:, 1:neighbor_count + 1].mean(axis=1)
        threshold = _robust_upper(mode_scores, 1.5)
        scores[rows] = mode_scores
        outlier_mask[rows] = mode_scores > threshold
        thresholds[str(mode)] = threshold
    return normalized, scores, outlier_mask, thresholds


def analyze_dataset(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    output_dir: str | Path | None = None,
    *,
    task_feature_prefixes: Sequence[str] = DEFAULT_TASK_FEATURE_PREFIXES,
    pca_variance_target: float = 0.90,
    dtw_resample_length: int = 60,
    dtw_window: int = 12,
    skip_dtw: bool = False,
    write_outputs: bool = False,
    verbose: bool = True,
    **legacy_options: Any,
) -> dict[str, Any]:
    """데이터셋을 분석하고 pass/validate/delete와 판정 근거를 반환한다."""
    # 이전 호출의 full_dbscan_features 인자는 더 이상 판정에 사용하지 않는다.
    legacy_options.pop("full_dbscan_features", None)
    if legacy_options:
        raise TypeError(f"지원하지 않는 옵션: {sorted(legacy_options)}")
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"데이터셋 디렉터리가 아닙니다: {dataset_dir}")
    if not 0.5 <= pca_variance_target <= 0.999:
        raise ValueError("pca_variance_target은 0.5~0.999 범위여야 합니다.")
    if dtw_resample_length < 2 or dtw_window < 1:
        raise ValueError("DTW resample length는 2 이상, window는 1 이상이어야 합니다.")

    dataset_info, joint_names = load_dataset_info(dataset_dir)
    parquet_files = discover_episode_files(dataset_dir)
    task_indices = np.array([
        idx for idx, name in enumerate(joint_names)
        if name.startswith(tuple(task_feature_prefixes))
    ], dtype=int)
    if not len(task_indices):
        raise ValueError(f"작업 feature를 찾지 못했습니다: {list(task_feature_prefixes)}")
    task_names = [joint_names[idx] for idx in task_indices]
    if verbose:
        print(f"데이터셋: {dataset_dir.name}")
        print(
            f"로봇: {dataset_info.get('robot_type', 'unknown')} | "
            f"Parquet: {len(parquet_files)}개 | state: {len(joint_names)}D | "
            f"task features: {len(task_indices)}D"
        )

    frame, trajectories, warnings, integrity_failures = extract_episode_features(
        parquet_files, joint_names, task_indices, verbose=verbose
    )
    if len(frame) < 5:
        raise ValueError("통계·군집 분석에는 유효한 에피소드가 최소 5개 필요합니다.")
    valid_ids = frame["episode_idx"].astype(int).to_numpy()
    all_ids = {_episode_id_from_path(path) for path in parquet_files}
    episode_count = len(all_ids)
    motion_matrix = np.stack(frame["motion_features"])
    full_motion_matrix = np.stack(frame["full_motion_features"])

    left_indices, right_indices = _hand_groups(joint_names, task_indices)
    hand_modes, left_share = _assign_hand_modes(
        motion_matrix, len(task_indices), left_indices, right_indices
    )
    frame["hand_mode"] = hand_modes
    frame["left_activity_share"] = left_share
    ordered_trajectories = [trajectories[int(idx)] for idx in valid_ids]
    total_path = []
    moving_fraction = []
    max_velocity = []
    max_acceleration = []
    max_jerk = []
    fps = float(dataset_info.get("fps") or 1.0)
    for trajectory in ordered_trajectories:
        task_trajectory = trajectory[:, task_indices]
        step = np.diff(task_trajectory, axis=0)
        acceleration_step = np.diff(task_trajectory, n=2, axis=0)
        jerk_step = np.diff(task_trajectory, n=3, axis=0)
        velocity = np.abs(step)
        total_path.append(float(np.sum(velocity)))
        moving_fraction.append(float(np.mean(np.max(velocity, axis=1) > 1e-4)))
        max_velocity.append(float(np.max(np.linalg.norm(step, axis=1))) * fps)
        max_acceleration.append(float(np.max(np.linalg.norm(acceleration_step, axis=1))) * fps ** 2)
        max_jerk.append(float(np.max(np.linalg.norm(jerk_step, axis=1))) * fps ** 3)
    total_path = np.asarray(total_path)
    moving_fraction = np.asarray(moving_fraction)
    max_velocity = np.asarray(max_velocity)
    max_acceleration = np.asarray(max_acceleration)
    max_jerk = np.asarray(max_jerk)
    impact_thresholds = {
        "max_velocity": _robust_upper(max_velocity, 3.0),
        "max_acceleration": _robust_upper(max_acceleration, 3.0),
        "max_jerk": _robust_upper(max_jerk, 3.0),
    }
    impact_mask = (
        (max_velocity > impact_thresholds["max_velocity"])
        & (max_acceleration > impact_thresholds["max_acceleration"])
        & (max_jerk > impact_thresholds["max_jerk"])
    )
    median_total_path = float(np.median(total_path))
    no_motion_mask = (
        (total_path < max(1e-6, 0.05 * median_total_path))
        | (moving_fraction < 0.10)
    )
    frame["total_joint_path"] = total_path
    frame["moving_fraction"] = moving_fraction
    frame["max_velocity"] = max_velocity
    frame["max_acceleration"] = max_acceleration
    frame["max_jerk"] = max_jerk
    temporal_matrix = _temporal_activity_features(
        ordered_trajectories, joint_names, task_indices
    )
    analysis_matrix = np.column_stack([motion_matrix, temporal_matrix])

    annotations, annotation_warnings = _annotation_profile(dataset_dir, valid_ids)
    warnings.extend(annotation_warnings)
    frame = frame.merge(annotations, on="episode_idx", how="left")
    available_annotations = frame.loc[frame["annotation_available"]]
    median_subtasks = (
        float(available_annotations["subtask_count"].median())
        if len(available_annotations) else None
    )
    median_skills = (
        float(available_annotations["skill_count"].median())
        if len(available_annotations) else None
    )
    mode_counts = Counter(hand_modes)
    # 서브태스크가 여러 개인 것을 가장 강한 복잡도 신호로 본다. 단일
    # subtask 안의 세부 skill 수만 많은 236형 태스크는 단순형으로 유지한다.
    complexity_score = 2 * int((median_subtasks or 0) > 1)
    complexity_score += int((median_skills or 0) >= 8)
    complexity_score += int(len(mode_counts) > 1)
    complexity_level = "complex" if complexity_score >= 3 else "simple"

    # 복잡 태스크에서 지배적인 반복 구조와 크게 다른 annotation은 작업 누락,
    # 재시도 또는 종료 단계 누락의 직접적인 검수 신호로 사용한다.
    annotation_structure_mask = np.zeros(len(frame), dtype=bool)
    if len(available_annotations) >= 5:
        signatures = list(zip(
            available_annotations["subtask_count"].astype(int),
            available_annotations["skill_count"].astype(int),
        ))
        signature_counts = Counter(signatures)
        dominant_signature, dominant_count = signature_counts.most_common(1)[0]
        rare_limit = max(2, int(np.floor(0.02 * len(available_annotations))))
        if dominant_count >= 0.5 * len(available_annotations):
            for row_idx, row in frame.iterrows():
                if not bool(row["annotation_available"]):
                    continue
                signature = (int(row["subtask_count"]), int(row["skill_count"]))
                skill_gap = abs(signature[1] - dominant_signature[1])
                subtask_gap = abs(signature[0] - dominant_signature[0])
                annotation_structure_mask[row_idx] = (
                    signature_counts[signature] <= rare_limit
                    and (skill_gap >= 2 or subtask_gap >= 1)
                )
    else:
        dominant_signature = None

    # 1. 길이: 일반 후보(1.5-IQR)와 강한 구조 신호(3-IQR)를 분리한다.
    lengths = frame["length"].to_numpy(dtype=float)
    q1_len, q3_len = np.quantile(lengths, [0.25, 0.75])
    iqr_len = float(q3_len - q1_len)
    lower_len = max(0.0, float(q1_len - 1.5 * iqr_len))
    upper_len = float(
        q3_len + 1.5 * iqr_len + max(2.0, 0.02 * float(np.median(lengths)))
    )
    severe_lower_len = max(0.0, float(q1_len - 3.0 * iqr_len))
    severe_upper_len = float(q3_len + 3.0 * iqr_len)
    length_mask = (lengths < lower_len) | (lengths > upper_len)
    severe_length_mask = (lengths < severe_lower_len) | (lengths > severe_upper_len)
    truncated_mask = lengths < 0.35 * float(np.median(lengths))
    length_outliers = valid_ids[length_mask].tolist()
    severe_length_outliers = valid_ids[severe_length_mask].tolist()
    truncated_outliers = valid_ids[truncated_mask].tolist()
    no_motion_outliers = valid_ids[no_motion_mask].tolist()

    # 2. 조인트 범위: 순간 max에만 의존하지 않도록 넓은 3-IQR 기준을 유지한다.
    min_matrix = np.stack(frame["min_joints"])[:, task_indices]
    max_matrix = np.stack(frame["max_joints"])[:, task_indices]
    q1_min, q3_min = np.quantile(min_matrix, [0.25, 0.75], axis=0)
    q1_max, q3_max = np.quantile(max_matrix, [0.25, 0.75], axis=0)
    span = np.ptp(np.vstack([min_matrix, max_matrix]), axis=0)
    min_iqr = np.maximum(q3_min - q1_min, np.maximum(0.05 * span, 1e-6))
    max_iqr = np.maximum(q3_max - q1_max, np.maximum(0.05 * span, 1e-6))
    lower_joint = q1_min - 3.0 * min_iqr
    upper_joint = q3_max + 3.0 * max_iqr
    low_excess = np.maximum(lower_joint - min_matrix, 0.0) / min_iqr
    high_excess = np.maximum(max_matrix - upper_joint, 0.0) / max_iqr
    joint_excess = np.maximum(low_excess, high_excess)
    joint_violation_count = np.sum(joint_excess > 0, axis=1)
    joint_mask = joint_violation_count >= 2
    severe_joint_mask = (
        (joint_violation_count >= max(3, int(np.ceil(0.20 * len(task_indices)))))
        & (np.max(joint_excess, axis=1) > 3.0)
    )
    joint_outliers = valid_ids[joint_mask].tolist()
    severe_joint_outliers = valid_ids[severe_joint_mask].tolist()

    # 3. 같은 손 사용 모드 안에서 Isolation Forest를 적용한다.
    isolation_scores, isolation_labels, isolation_thresholds = _mode_aware_isolation(
        analysis_matrix, hand_modes
    )
    isolation_mask = isolation_labels == -1
    isolation_outliers = valid_ids[isolation_mask].tolist()
    frame["isolation_score"] = isolation_scores

    # 4. 설명분산 기반 PCA + 모드별 HDBSCAN. noise는 실패가 아닌 희귀 수행이다.
    cluster_labels, cluster_probabilities, cluster_mask, cluster_embedding, cluster_diagnostics = (
        _mode_aware_hdbscan(analysis_matrix, hand_modes, pca_variance_target)
    )
    cluster_outliers = valid_ids[cluster_mask].tolist()
    frame["cluster_label"] = cluster_labels
    frame["cluster_probability"] = cluster_probabilities

    # 5. DTW도 서로 다른 손 사용 모드끼리 직접 비교하지 않는다.
    trajectory_list = [trajectory[:, task_indices] for trajectory in ordered_trajectories]
    if skip_dtw:
        normalized_trajectories = [resample_trajectory(t - t[0], dtw_resample_length) for t in trajectory_list]
        dtw_scores = np.full(len(valid_ids), np.nan)
        dtw_mask = np.zeros(len(valid_ids), dtype=bool)
        dtw_thresholds: dict[str, float] = {}
    else:
        normalized_trajectories, dtw_scores, dtw_mask, dtw_thresholds = _mode_aware_dtw(
            trajectory_list, hand_modes, dtw_resample_length, dtw_window, verbose
        )
    dtw_outliers = valid_ids[dtw_mask].tolist()
    frame["dtw_score"] = dtw_scores

    # 6. annotation 누락은 삭제 사유가 아니라 메타데이터 검수 사유다.
    annotation_issue_mask = (
        frame["annotation_available"].astype(bool)
        & ((frame["annotation_coverage"] < 0.98) | (frame["annotation_coverage"] > 1.02))
    ).to_numpy()
    annotation_issue_outliers = valid_ids[annotation_issue_mask].tolist()
    annotation_structure_outliers = valid_ids[annotation_structure_mask].tolist()
    impact_outliers = valid_ids[impact_mask].tolist()

    signal_sets = {
        "length": set(length_outliers),
        "joint_range": set(joint_outliers),
        "motion_isolation": set(isolation_outliers),
        "mode_cluster": set(cluster_outliers),
        "trajectory_dtw": set(dtw_outliers),
        "annotation_integrity": set(annotation_issue_outliers),
        "annotation_structure": set(annotation_structure_outliers),
        "no_motion": set(no_motion_outliers),
        "severe_truncation": set(truncated_outliers),
        "impact": set(impact_outliers),
    }
    evidence_counts = {
        int(idx): sum(int(idx) in values for values in signal_sets.values())
        for idx in valid_ids
    }

    # 자동 삭제는 명백한 파일 무결성 실패 또는 강한 구조 신호가 다른 신호와
    # 함께 발생한 경우로 제한한다. 클러스터 noise 단독 삭제는 금지한다.
    delete_labels = (
        set(integrity_failures) | set(no_motion_outliers) | set(truncated_outliers)
    )
    for idx in valid_ids:
        episode = int(idx)
        corroborating = sum(episode in signal_sets[name] for name in (
            "motion_isolation", "mode_cluster", "trajectory_dtw"
        ))
        structural = episode in set(severe_length_outliers) or episode in set(severe_joint_outliers)
        annotation_broken = episode in set(annotation_issue_outliers)
        simple_truncation = (
            complexity_level == "simple"
            and episode in signal_sets["length"]
            and episode in signal_sets["motion_isolation"]
        )
        if (
            (structural and corroborating >= 1)
            or (annotation_broken and evidence_counts[episode] >= 3)
            or simple_truncation
        ):
            delete_labels.add(episode)

    structural_candidates = (
        signal_sets["length"] | signal_sets["joint_range"]
        | signal_sets["annotation_integrity"]
    )
    if complexity_level == "simple":
        # 단순 동작에서는 전체 운동 요약 자체가 강한 신호다. 군집/DTW 단독
        # noise는 약한 신호이므로 두 방법이 동의할 때만 후보에 더한다.
        behavioral_candidates = signal_sets["motion_isolation"] | (
            signal_sets["mode_cluster"] & signal_sets["trajectory_dtw"]
        )
    else:
        annotation_coverage_ratio = float(frame["annotation_available"].mean())
        if annotation_coverage_ratio >= 0.80:
            # annotation이 충분하면 반복 횟수/단계 구조를 모션 희귀도보다 우선한다.
            behavioral_candidates = signal_sets["annotation_structure"]
            structural_candidates = (
                signal_sets["annotation_integrity"]
                | set(severe_length_outliers) | set(severe_joint_outliers)
            )
        else:
            # annotation이 없는 복잡 태스크는 세 계열이 모두 동의해야 후보가 된다.
            behavioral_candidates = (
                signal_sets["motion_isolation"] & signal_sets["mode_cluster"]
                & signal_sets["trajectory_dtw"]
            )
    # 세 순간 충격 지표가 데이터셋별 3-IQR을 모두 넘을 때만 보수적으로 검수한다.
    behavioral_candidates |= signal_sets["impact"]
    all_candidates = structural_candidates | behavioral_candidates
    validate_labels = all_candidates - delete_labels
    pass_labels = all_ids - delete_labels - validate_labels
    final_recommendations = delete_labels | validate_labels

    result: dict[str, Any] = {
        "DATASET_DIR": dataset_dir,
        "OUTPUT_DIR": Path(output_dir).expanduser().resolve() if output_dir else dataset_dir / "analysis_results",
        "dataset_info": dataset_info,
        "episode_count": episode_count,
        "valid_episode_count": len(valid_ids),
        "joint_names": joint_names,
        "task_feature_indices": task_indices,
        "task_joint_names": task_names,
        "motion_block_names": list(MOTION_BLOCK_NAMES),
        "df_summary": frame,
        "all_joint_trajectories": trajectories,
        "motion_feature_matrix": motion_matrix,
        "temporal_activity_feature_matrix": temporal_matrix,
        "analysis_feature_matrix": analysis_matrix,
        "full_motion_feature_matrix": full_motion_matrix,
        "hand_modes": hand_modes,
        "hand_mode_counts": dict(mode_counts),
        "left_activity_share": left_share,
        "total_joint_path": total_path,
        "moving_fraction": moving_fraction,
        "impact_thresholds": impact_thresholds,
        "impact_outliers": impact_outliers,
        "complexity_profile": {
            "level": complexity_level,
            "score": complexity_score,
            "median_subtask_count": median_subtasks,
            "median_skill_count": median_skills,
            "hand_mode_counts": dict(mode_counts),
            "dominant_annotation_signature": dominant_signature,
        },
        "length_bounds": {"lower": lower_len, "upper": upper_len,
                          "severe_lower": severe_lower_len, "severe_upper": severe_upper_len},
        "length_outliers": length_outliers,
        "severe_length_outliers": severe_length_outliers,
        "truncated_outliers": truncated_outliers,
        "no_motion_outliers": no_motion_outliers,
        "joint_min_matrix": min_matrix,
        "joint_max_matrix": max_matrix,
        "joint_excess": joint_excess,
        "joint_violation_count": joint_violation_count,
        "joint_range_outliers": joint_outliers,
        "severe_joint_outliers": severe_joint_outliers,
        "isolation_scores": isolation_scores,
        "isolation_thresholds": isolation_thresholds,
        "isolation_outliers": isolation_outliers,
        "cluster_labels": cluster_labels,
        "cluster_probabilities": cluster_probabilities,
        "cluster_embedding": cluster_embedding,
        "cluster_diagnostics": cluster_diagnostics,
        "cluster_outliers": cluster_outliers,
        "normalized_trajectories": normalized_trajectories,
        "dtw_scores": dtw_scores,
        "dtw_thresholds": dtw_thresholds,
        "dtw_outliers": dtw_outliers,
        "annotation_issue_outliers": annotation_issue_outliers,
        "annotation_structure_outliers": annotation_structure_outliers,
        "integrity_failures": integrity_failures,
        "warnings": warnings,
        "signal_sets": signal_sets,
        "evidence_counts": evidence_counts,
        "all_candidates": all_candidates,
        "all_outliers": final_recommendations,
        "delete_labels": delete_labels,
        "validate_labels": validate_labels,
        "pass_labels": pass_labels,
        "analysis_parameters": {
            "task_feature_prefixes": list(task_feature_prefixes),
            "pca_variance_target": pca_variance_target,
            "hand_mode_boundaries": [0.35, 0.65],
            "cluster_algorithm": "mode-aware HDBSCAN",
            "dtw_resample_length": dtw_resample_length,
            "dtw_window": dtw_window,
            "skip_dtw": skip_dtw,
        },
    }
    if write_outputs:
        save_analysis_outputs(result)
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def save_analysis_outputs(result: dict[str, Any]) -> None:
    output_dir = Path(result["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "delete": sorted(result["delete_labels"]),
        "validate": sorted(result["validate_labels"]),
        "pass": sorted(result["pass_labels"]),
    }
    with (output_dir / "episode_outlier_labels.json").open("w", encoding="utf-8") as file:
        json.dump(labels, file, ensure_ascii=False, indent=2)

    summary = {
        "schema_version": 2,
        "dataset": result["DATASET_DIR"].name,
        "robot_type": result["dataset_info"].get("robot_type"),
        "episode_count": result["episode_count"],
        "valid_episode_count": result["valid_episode_count"],
        "state_dimension": len(result["joint_names"]),
        "task_feature_names": result["task_joint_names"],
        "complexity_profile": result["complexity_profile"],
        "parameters": result["analysis_parameters"],
        "length_bounds": result["length_bounds"],
        "isolation_thresholds_by_mode": result["isolation_thresholds"],
        "dtw_thresholds_by_mode": result["dtw_thresholds"],
        "impact_thresholds": result["impact_thresholds"],
        "cluster_diagnostics": result["cluster_diagnostics"],
        "signals": {name: sorted(values) for name, values in result["signal_sets"].items()},
        "severe_signals": {
            "length": result["severe_length_outliers"],
            "severe_truncation": result["truncated_outliers"],
            "no_motion": result["no_motion_outliers"],
            "joint_range": result["severe_joint_outliers"],
            "file_integrity": result["integrity_failures"],
        },
        "labels": labels,
        "label_counts": {label: len(indices) for label, indices in labels.items()},
        "warnings": result["warnings"],
    }
    with (output_dir / "outlier_summary.json").open("w", encoding="utf-8") as file:
        json.dump(_json_value(summary), file, ensure_ascii=False, indent=2)

    frame = result["df_summary"][[
        "episode_idx", "length", "hand_mode", "left_activity_share",
        "total_joint_path", "moving_fraction",
        "max_velocity", "max_acceleration", "max_jerk",
        "subtask_count", "skill_count", "isolation_score", "cluster_label",
        "cluster_probability", "dtw_score",
    ]].copy()
    all_ids = pd.DataFrame({"episode_idx": sorted(
        result["pass_labels"] | result["validate_labels"] | result["delete_labels"]
    )})
    table = all_ids.merge(frame, on="episode_idx", how="left")
    ids = table["episode_idx"]
    for name, values in result["signal_sets"].items():
        table[f"{name}_outlier"] = ids.isin(values)
    table["evidence_count"] = ids.map(result["evidence_counts"]).fillna(0).astype(int)
    label_lookup = {
        **{idx: "delete" for idx in result["delete_labels"]},
        **{idx: "validate" for idx in result["validate_labels"]},
        **{idx: "pass" for idx in result["pass_labels"]},
    }
    table["label"] = ids.map(label_lookup)
    table["reasons"] = [
        ",".join(name for name, values in result["signal_sets"].items() if int(idx) in values)
        or ("file_integrity" if int(idx) in result["integrity_failures"] else "")
        for idx in ids
    ]
    table.to_csv(output_dir / "outlier_results.csv", index=False)
    save_analysis_plot(result, output_dir / "episode_outlier_analysis.png")


def save_analysis_plot(result: dict[str, Any], output_path: Path) -> None:
    frame = result["df_summary"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(frame["length"], bins=15, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(result["length_bounds"]["lower"], color="red", linestyle="--")
    axes[0].axvline(result["length_bounds"]["upper"], color="red", linestyle="--")
    axes[0].set(title="Episode length", xlabel="Frames", ylabel="Count")

    embedding = result["cluster_embedding"]
    if embedding.shape[1] >= 2:
        colors = [
            "red" if int(idx) in result["delete_labels"] else
            "orange" if int(idx) in result["validate_labels"] else "green"
            for idx in frame["episode_idx"]
        ]
        axes[1].scatter(embedding[:, 0], embedding[:, 1], c=colors, s=42,
                        edgecolors="black", alpha=0.8)
        axes[1].set(xlabel="Adaptive PC1", ylabel="Adaptive PC2")
    axes[1].set_title("Mode-aware motion quality candidates")
    axes[1].grid(True, linestyle=":")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def print_analysis_report(result: dict[str, Any]) -> None:
    print("\n" + "=" * 58)
    print("                 범용 데이터 품질 분석")
    print("=" * 58)
    print(f"■ 데이터셋: {result['DATASET_DIR'].name}")
    print(f"■ 에피소드: {result['episode_count']}개 (분석 가능 {result['valid_episode_count']}개)")
    profile = result["complexity_profile"]
    print(
        f"■ 복잡도: {profile['level']} | subtask 중앙값={profile['median_subtask_count']} | "
        f"skill 중앙값={profile['median_skill_count']}"
    )
    print(f"■ 손 사용 모드: {profile['hand_mode_counts']}")
    print(f"■ 길이 후보: {result['length_outliers']}")
    print(f"■ 무동작/정지: {result['no_motion_outliers']}")
    print(f"■ 심각한 잘림: {result['truncated_outliers']}")
    print(f"■ 조인트 범위 후보: {result['joint_range_outliers']}")
    print(f"■ 모드별 Isolation 후보: {result['isolation_outliers']}")
    print(f"■ 모드별 HDBSCAN 후보: {result['cluster_outliers']}")
    print(f"■ 모드별 DTW 후보: {result['dtw_outliers']}")
    print(f"■ Annotation 무결성 후보: {result['annotation_issue_outliers']}")
    print(f"■ Annotation 구조 후보: {result['annotation_structure_outliers']}")
    print(f"■ 순간 충격 후보: {result['impact_outliers']}")
    print("-" * 58)
    print(f"★ delete ({len(result['delete_labels'])}): {sorted(result['delete_labels'])}")
    print(f"★ validate ({len(result['validate_labels'])}): {sorted(result['validate_labels'])}")
    print(f"★ pass ({len(result['pass_labels'])})")
    print("=" * 58)
    print(f"결과 저장 위치: {result['OUTPUT_DIR']}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LeRobot 범용 에피소드 품질 분석")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--task-feature-prefixes", nargs="+", default=list(DEFAULT_TASK_FEATURE_PREFIXES)
    )
    parser.add_argument("--pca-variance-target", type=float, default=0.90)
    parser.add_argument("--dtw-resample-length", type=int, default=60)
    parser.add_argument("--dtw-window", type=int, default=12)
    parser.add_argument("--skip-dtw", action="store_true")
    parser.add_argument("--no-write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_argument_parser().parse_args(argv)
    result = analyze_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        task_feature_prefixes=args.task_feature_prefixes,
        pca_variance_target=args.pca_variance_target,
        dtw_resample_length=args.dtw_resample_length,
        dtw_window=args.dtw_window,
        skip_dtw=args.skip_dtw,
        write_outputs=not args.no_write_results,
        verbose=not args.quiet,
    )
    print_analysis_report(result)
    return result


if __name__ == "__main__":
    try:
        main()
    finally:
        elapsed = time.perf_counter() - PROGRAM_STARTED_AT
        print(f"총 실행 시간: {elapsed:.2f}초")
