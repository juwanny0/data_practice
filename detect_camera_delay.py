"""LeRobot 영상과 관절 상태 사이의 카메라 지연을 검출한다.

MP4의 PTS나 프레임 수가 정상이어도 영상 내용은 로봇 상태보다 늦을 수 있다.
이 프로그램은 저해상도 영상의 프레임간 변화량과 대응 관절의 속도 신호를
교차상관하여 에피소드/카메라별 상대 지연을 추정한다.

양수 ``delay_frames``는 영상이 관절 상태보다 늦다는 뜻이다. 예를 들어
15 FPS에서 +30 프레임은 관절이 움직인 약 2초 뒤에 영상에서 그 움직임이
관측된다는 뜻이다.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from execution_logging import start_automatic_logging

if __name__ == "__main__":
    start_automatic_logging()

import av
import numpy as np
import pandas as pd


PROGRAM_STARTED_AT = time.perf_counter()

# 이 값보다 작은 관절 속도 변화는 실제 동작보다 센서 양자화/미세 진동일
# 가능성이 크다. 확정 지연(216)의 값은 약 0.027 이상인 반면, 확인된
# 녹화 오류(236-15)는 약 0.000207이다.
MIN_MEANINGFUL_JOINT_MOTION_STD = 1e-3
MIN_WRIST_PEAK_CORRELATION = 0.60
MIN_WRIST_CORRELATION_GAIN = 0.10
MIN_WRIST_PEAK_PROMINENCE = 0.02
MIN_FIXED_PEAK_PROMINENCE = 0.05


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = (
    BASE_DIR
    / "Task_000216_Sub3_Put_Carrot_Plate_L2R_Speed40_Lift18_KBK_NJW_fail_example_lerobot"
)


@dataclass
class DelayResult:
    episode_idx: int
    camera: str
    camera_group: str
    status: str
    is_delay_outlier: bool
    delay_frames: int | None
    delay_seconds: float | None
    peak_correlation: float | None
    zero_lag_correlation: float | None
    correlation_gain: float | None
    peak_prominence: float | None
    window_delay_median: float | None
    window_delay_mad: float | None
    reliable_window_count: int
    video_frame_count: int
    decoded_frame_count: int
    parquet_row_count: int
    frame_count_delta: int
    video_duration_seconds: float | None
    analyzed_video_seconds: float | None
    pts_step_min: float | None
    pts_step_max: float | None
    nonpositive_pts_steps: int
    corrupt_frame_count: int
    video_motion_std: float | None
    joint_motion_std: float | None
    cache_hit: bool
    reason: str


def _finite_float(value: float | np.floating | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def load_dataset_info(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"메타데이터를 찾을 수 없습니다: {info_path}")
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)
    if not info.get("fps"):
        raise ValueError("meta/info.json에 유효한 fps가 없습니다.")
    return info


def discover_episode_files(dataset_dir: Path) -> list[Path]:
    files = sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError("data/chunk-*/episode_*.parquet을 찾을 수 없습니다.")
    return files


def discover_video_keys(info: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    )


def episode_index_from_path(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def _flat_state_columns(columns: Iterable[str]) -> list[str]:
    candidates = [column for column in columns if column.startswith("observation.state")]

    def sort_key(column: str) -> tuple[int, str]:
        suffix = column.rsplit("_", 1)[-1]
        return (int(suffix), column) if suffix.isdigit() else (10**9, column)

    return sorted(candidates, key=sort_key)


def load_state(path: Path) -> tuple[np.ndarray, int]:
    # 대부분의 LeRobot Parquet에는 큰 action 등 다른 컬럼도 포함된다.
    # 필요한 두 컬럼만 우선 읽고, 구형 flat-state 스키마만 전체 읽기로 fallback한다.
    try:
        frame = pd.read_parquet(
            path, columns=["observation.state", "episode_index"]
        )
    except (KeyError, ValueError):
        frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"빈 Parquet: {path}")
    if "observation.state" in frame:
        state = np.asarray(frame["observation.state"].tolist(), dtype=np.float64)
    else:
        columns = _flat_state_columns(frame.columns)
        if not columns:
            raise ValueError(f"observation.state 컬럼이 없습니다: {path}")
        state = frame[columns].to_numpy(dtype=np.float64)
    if state.ndim != 2 or len(state) < 3 or not np.isfinite(state).all():
        raise ValueError(f"유효하지 않은 observation.state: {path}, shape={state.shape}")
    episode_idx = (
        int(frame["episode_index"].iloc[0])
        if "episode_index" in frame else episode_index_from_path(path)
    )
    return state, episode_idx


def joint_indices_for_camera(camera: str, joint_names: Sequence[str]) -> tuple[np.ndarray, str]:
    """카메라 움직임을 직접 유발하는 팔 관절만 선택한다.

    그리퍼는 카메라 자세를 바꾸지 않으며 물체 반응과의 자연스러운 시간차를
    카메라 지연으로 오인하게 하므로 제외한다.
    """
    lower = camera.lower()
    left = "left" in lower or "_l_" in lower
    right = "right" in lower or "_r_" in lower
    wrist = "wrist" in lower or "hand" in lower

    if wrist and left:
        prefixes = ("arm_l_", "left_arm")
        group = "left_wrist"
    elif wrist and right:
        prefixes = ("arm_r_", "right_arm")
        group = "right_wrist"
    else:
        # 고정 head 카메라는 양팔이 보이지만 영상 움직임과 관절 움직임의 관계가
        # 손목 카메라보다 간접적이다. 진단값은 계산하되 더 엄격하게 판정한다.
        prefixes = (
            "arm_l_", "arm_r_", "left_arm", "right_arm",
        )
        group = "head_or_fixed"
    indices = np.array([
        idx for idx, name in enumerate(joint_names)
        if name.lower().startswith(prefixes)
    ], dtype=int)
    if not len(indices):
        raise ValueError(f"{camera}에 대응하는 관절을 찾지 못했습니다.")
    return indices, group


def find_video_path(dataset_dir: Path, video_key: str, episode_idx: int) -> Path:
    matches = list(dataset_dir.glob(
        f"videos/chunk-*/{video_key}/episode_{episode_idx:06d}.mp4"
    ))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"영상이 없거나 중복됩니다: {video_key}, episode {episode_idx}, matches={len(matches)}"
        )
    return matches[0]


def prepare_gray_frame(video_frame: av.VideoFrame, width: int, height: int) -> np.ndarray:
    """조명 변화는 제거하고 구조적 영상 변화만 남긴다."""
    gray = video_frame.reformat(
        width=width, height=height, format="gray"
    ).to_ndarray().astype(np.float32)
    return (gray - float(np.mean(gray))) / (float(np.std(gray)) + 5.0)


def robust_frame_motion(current: np.ndarray, previous: np.ndarray) -> float:
    """국소적인 사람/물체 움직임의 큰 픽셀 차이를 제한한다."""
    difference = np.abs(current - previous)
    flat = difference.ravel()
    cutoff_index = int(0.90 * (len(flat) - 1))
    cutoff = float(np.partition(flat, cutoff_index)[cutoff_index])
    return float(np.mean(np.minimum(difference, cutoff)))


def decode_video_motion(
    video_path: Path,
    *,
    width: int,
    height: int,
    max_frames: int,
    decoder_threads: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """앞부분만 디코딩하고 저해상도 grayscale frame difference를 만든다."""
    previous: np.ndarray | None = None
    motion: list[float] = []
    pts: list[float] = []
    frame_count = 0
    corrupt_count = 0
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = decoder_threads
        reported_frame_count = int(stream.frames or 0)
        reported_duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None and stream.time_base is not None else None
        )
        for video_frame in container.decode(stream):
            corrupt_count += int(video_frame.is_corrupt)
            gray = prepare_gray_frame(video_frame, width, height)
            if previous is not None:
                motion.append(robust_frame_motion(gray, previous))
            previous = gray
            pts.append(float(video_frame.time) if video_frame.time is not None else np.nan)
            frame_count += 1
            if frame_count >= max_frames:
                break
    finally:
        container.close()

    pts_array = np.asarray(pts, dtype=float)
    pts_steps = np.diff(pts_array)
    metadata = {
        "reported_frame_count": reported_frame_count,
        "decoded_frame_count": frame_count,
        "reported_duration": reported_duration,
        "analyzed_duration": (
            float(pts_array[-1] - pts_array[0]) if len(pts_array) > 1 else None
        ),
        "pts_step_min": _finite_float(np.nanmin(pts_steps)) if len(pts_steps) else None,
        "pts_step_max": _finite_float(np.nanmax(pts_steps)) if len(pts_steps) else None,
        "nonpositive_pts_steps": int(np.sum(pts_steps <= 0)),
        "corrupt_count": corrupt_count,
    }
    return np.asarray(motion, dtype=np.float64), metadata


def _motion_cache_signature(
    video_path: Path, *, width: int, height: int, max_frames: int
) -> dict[str, int]:
    stat = video_path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "width": width,
        "height": height,
        "max_frames": max_frames,
        "cache_schema": 2,
    }


def _load_motion_cache(
    cache_path: Path | None, signature: dict[str, int]
) -> tuple[np.ndarray, dict[str, Any]] | None:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            if json.loads(str(cached["signature"].item())) != signature:
                return None
            metadata = json.loads(str(cached["metadata"].item()))
            return cached["motion"].astype(np.float64), metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _save_motion_cache(
    cache_path: Path | None,
    signature: dict[str, int],
    motion: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        motion=motion,
        metadata=json.dumps(metadata),
        signature=json.dumps(signature),
    )
    temporary.replace(cache_path)


def _decoded_metadata(
    *,
    reported_frame_count: int,
    reported_duration: float | None,
    pts: Sequence[float],
    corrupt_count: int,
) -> dict[str, Any]:
    pts_array = np.asarray(pts, dtype=float)
    pts_steps = np.diff(pts_array)
    return {
        "reported_frame_count": reported_frame_count,
        "decoded_frame_count": len(pts),
        "reported_duration": reported_duration,
        "analyzed_duration": (
            float(pts_array[-1] - pts_array[0]) if len(pts_array) > 1 else None
        ),
        "pts_step_min": _finite_float(np.nanmin(pts_steps)) if len(pts_steps) else None,
        "pts_step_max": _finite_float(np.nanmax(pts_steps)) if len(pts_steps) else None,
        "nonpositive_pts_steps": int(np.sum(pts_steps <= 0)),
        "corrupt_count": corrupt_count,
    }


def load_or_decode_video_motion_staged(
    video_path: Path,
    *,
    width: int,
    height: int,
    prefix_frames: int,
    full_frames: int,
    decoder_threads: int,
    prefix_cache_path: Path | None,
    full_cache_path: Path | None,
    needs_full_verification: Any,
) -> tuple[np.ndarray, dict[str, Any], bool, bool]:
    """초기 판정 뒤 필요할 때 같은 디코더로 끝까지 이어 읽는다.

    반환값은 motion, metadata, cache_hit, full_verification 순서다.
    """
    prefix_signature = _motion_cache_signature(
        video_path, width=width, height=height, max_frames=prefix_frames
    )
    cached_prefix = _load_motion_cache(prefix_cache_path, prefix_signature)
    if cached_prefix is not None:
        prefix_motion, prefix_meta = cached_prefix
        if not needs_full_verification(prefix_motion):
            return prefix_motion, prefix_meta, True, False
        full_signature = _motion_cache_signature(
            video_path, width=width, height=height, max_frames=full_frames
        )
        cached_full = _load_motion_cache(full_cache_path, full_signature)
        if cached_full is not None:
            return cached_full[0], cached_full[1], True, True
        # 캐시가 초기 구간만 존재하는 중단/설정변경 상황은 전체를 새로 읽는다.
        motion, metadata = decode_video_motion(
            video_path, width=width, height=height, max_frames=full_frames,
            decoder_threads=decoder_threads,
        )
        _save_motion_cache(full_cache_path, full_signature, motion, metadata)
        return motion, metadata, False, True

    previous: np.ndarray | None = None
    motion_values: list[float] = []
    pts: list[float] = []
    corrupt_count = 0
    continue_to_full = False
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = decoder_threads
        reported_frame_count = int(stream.frames or 0)
        reported_duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None and stream.time_base is not None else None
        )
        for video_frame in container.decode(stream):
            corrupt_count += int(video_frame.is_corrupt)
            gray = prepare_gray_frame(video_frame, width, height)
            if previous is not None:
                motion_values.append(robust_frame_motion(gray, previous))
            previous = gray
            pts.append(float(video_frame.time) if video_frame.time is not None else np.nan)
            decoded_count = len(pts)
            if decoded_count == prefix_frames:
                prefix_motion = np.asarray(motion_values, dtype=np.float64)
                prefix_meta = _decoded_metadata(
                    reported_frame_count=reported_frame_count,
                    reported_duration=reported_duration,
                    pts=pts,
                    corrupt_count=corrupt_count,
                )
                _save_motion_cache(
                    prefix_cache_path, prefix_signature, prefix_motion, prefix_meta
                )
                continue_to_full = bool(needs_full_verification(prefix_motion))
                if not continue_to_full:
                    return prefix_motion, prefix_meta, False, False
            if decoded_count >= full_frames:
                break
    finally:
        container.close()

    motion = np.asarray(motion_values, dtype=np.float64)
    metadata = _decoded_metadata(
        reported_frame_count=reported_frame_count,
        reported_duration=reported_duration,
        pts=pts,
        corrupt_count=corrupt_count,
    )
    full_signature = _motion_cache_signature(
        video_path, width=width, height=height, max_frames=full_frames
    )
    _save_motion_cache(full_cache_path, full_signature, motion, metadata)
    return motion, metadata, False, continue_to_full


def joint_motion_signal(state: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(state[:, indices], axis=0), axis=1)


def smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.astype(float, copy=True)
    window = min(window, len(signal))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(signal, kernel, mode="same")


def lag_correlation_curve(
    joint_motion: np.ndarray,
    video_motion: np.ndarray,
    *,
    max_lag_frames: int,
    min_overlap_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """양수 lag일 때 video[lag:]를 joint[:-lag]와 비교한다."""
    lags = np.arange(-max_lag_frames, max_lag_frames + 1, dtype=int)
    correlations = np.full(len(lags), np.nan, dtype=float)
    for position, lag in enumerate(lags):
        if lag >= 0:
            overlap = min(len(joint_motion), len(video_motion) - lag)
            if overlap <= 0:
                continue
            joint_part = joint_motion[:overlap]
            video_part = video_motion[lag: lag + overlap]
        else:
            lead = -lag
            overlap = min(len(joint_motion) - lead, len(video_motion))
            if overlap <= 0:
                continue
            joint_part = joint_motion[lead: lead + overlap]
            video_part = video_motion[:overlap]
        finite = np.isfinite(joint_part) & np.isfinite(video_part)
        if finite.sum() < min_overlap_frames:
            continue
        joint_part = joint_part[finite]
        video_part = video_part[finite]
        if np.std(joint_part) < 1e-12 or np.std(video_part) < 1e-12:
            continue
        correlations[position] = float(np.corrcoef(joint_part, video_part)[0, 1])
    return lags, correlations


def estimate_lag(
    joint_motion: np.ndarray,
    video_motion: np.ndarray,
    *,
    max_lag_frames: int,
    smoothing_frames: int,
    min_overlap_frames: int,
) -> dict[str, Any]:
    joint_smoothed = smooth_signal(joint_motion, smoothing_frames)
    video_smoothed = smooth_signal(video_motion, smoothing_frames)
    lags, correlations = lag_correlation_curve(
        joint_smoothed,
        video_smoothed,
        max_lag_frames=max_lag_frames,
        min_overlap_frames=min_overlap_frames,
    )
    if not np.isfinite(correlations).any():
        raise ValueError("유효한 시차 상관을 계산할 수 없습니다.")
    peak_position = int(np.nanargmax(correlations))
    peak_lag = int(lags[peak_position])
    peak_correlation = float(correlations[peak_position])
    zero_position = int(np.flatnonzero(lags == 0)[0])
    zero_correlation = float(correlations[zero_position])

    # 넓은 correlation peak 자체를 불확실로 오인하지 않도록 최고점 주변 ±3은
    # prominence 비교에서 제외한다.
    competitors = correlations.copy()
    competitors[max(0, peak_position - 3): peak_position + 4] = np.nan
    runner_up = float(np.nanmax(competitors)) if np.isfinite(competitors).any() else np.nan
    prominence = peak_correlation - runner_up if np.isfinite(runner_up) else np.nan
    return {
        "delay_frames": peak_lag,
        "peak_correlation": peak_correlation,
        "zero_lag_correlation": zero_correlation,
        "correlation_gain": peak_correlation - zero_correlation,
        "peak_prominence": prominence,
        "joint_smoothed": joint_smoothed,
        "video_smoothed": video_smoothed,
    }


def estimate_window_consistency(
    joint_motion: np.ndarray,
    video_motion: np.ndarray,
    *,
    max_lag_frames: int,
    smoothing_frames: int,
    min_overlap_frames: int,
    windows: int = 3,
) -> tuple[float | None, float | None, int]:
    """겹치는 시간 창에서 독립적으로 시차를 재추정한다."""
    length = min(len(joint_motion), len(video_motion))
    if length < max(3 * min_overlap_frames, 90):
        return None, None, 0
    window_width = max(length // 2, min_overlap_frames + 2 * max_lag_frames)
    starts = np.linspace(0, max(0, length - window_width), windows).round().astype(int)
    estimates: list[int] = []
    for start in np.unique(starts):
        end = min(length, start + window_width)
        try:
            estimate = estimate_lag(
                joint_motion[start:end], video_motion[start:end],
                max_lag_frames=max_lag_frames,
                smoothing_frames=smoothing_frames,
                min_overlap_frames=min_overlap_frames,
            )
        except ValueError:
            continue
        if estimate["peak_correlation"] >= 0.45:
            estimates.append(int(estimate["delay_frames"]))
    if not estimates:
        return None, None, 0
    median = float(np.median(estimates))
    mad = float(np.median(np.abs(np.asarray(estimates) - median)))
    return median, mad, len(estimates)


def classify_delay(
    estimate: dict[str, Any],
    *,
    camera_group: str,
    fps: float,
    max_lag_frames: int,
    min_delay_seconds: float,
    min_peak_correlation: float,
    min_correlation_gain: float,
    window_median: float | None,
    window_mad: float | None,
    reliable_windows: int,
    video_motion_std: float,
    joint_motion_std: float,
    global_joint_motion_std: float,
) -> tuple[bool, str, str]:
    delay = abs(int(estimate["delay_frames"]))
    signed_delay = int(estimate["delay_frames"])
    delay_limit = max(2, int(math.ceil(min_delay_seconds * fps)))
    peak = float(estimate["peak_correlation"])
    gain = float(estimate["correlation_gain"])
    prominence = float(estimate["peak_prominence"])

    if joint_motion_std < MIN_MEANINGFUL_JOINT_MOTION_STD:
        if (
            global_joint_motion_std < MIN_MEANINGFUL_JOINT_MOTION_STD
            and video_motion_std >= 1e-3
        ):
            return False, "recording_error", (
                "영상은 움직이지만 로봇 전체 관절 상태가 거의 고정됨"
                f"(global_joint_std={global_joint_motion_std:.6g})"
            )
        return False, "insufficient_motion", "대응 관절 움직임이 부족함"
    if video_motion_std < 1e-3:
        return False, "insufficient_motion", "영상 움직임이 부족함"
    if camera_group == "head_or_fixed":
        # 고정 카메라에는 물체/반대팔 움직임이 섞이므로 강한 증거만 허용한다.
        required_peak = max(min_peak_correlation, 0.65)
        required_gain = max(min_correlation_gain, 0.20)
        required_prominence = MIN_FIXED_PEAK_PROMINENCE
    else:
        required_peak = max(min_peak_correlation, MIN_WRIST_PEAK_CORRELATION)
        required_gain = max(min_correlation_gain, MIN_WRIST_CORRELATION_GAIN)
        required_prominence = MIN_WRIST_PEAK_PROMINENCE
    # 구간별 결과는 신뢰도 해석용으로 보존하지만 hard gate로 쓰지 않는다.
    # grasp/contact처럼 일부 구간의 시각 변화가 관절 속도와 약하게 연결되는
    # 에피소드에서는 전체 신호의 명확한 고정 시차마저 놓칠 수 있기 때문이다.
    is_outlier = (
        signed_delay > 0
        and delay >= delay_limit
        # 탐색 범위 끝의 최고점은 실제 피크가 범위 밖일 수 있어 확정하지 않는다.
        and signed_delay < max_lag_frames
        and peak >= required_peak
        and gain >= required_gain
        and prominence >= required_prominence
    )
    if is_outlier:
        direction = "영상 지연" if estimate["delay_frames"] > 0 else "영상 선행"
        return True, "delay", (
            f"{direction} {estimate['delay_frames']:+d}프레임; "
            f"peak={peak:.3f}, zero={estimate['zero_lag_correlation']:.3f}, gain={gain:.3f}"
        )
    return False, "pass", (
        f"판정 기준 미충족; lag={estimate['delay_frames']:+d}, "
        f"peak={peak:.3f}, gain={gain:.3f}, prominence={prominence:.3f}"
    )


def analyze_camera_episode(
    *,
    dataset_dir: Path,
    state: np.ndarray,
    episode_idx: int,
    video_key: str,
    joint_names: Sequence[str],
    fps: float,
    width: int,
    height: int,
    max_lag_seconds: float,
    min_delay_seconds: float,
    smoothing_frames: int,
    min_peak_correlation: float,
    min_correlation_gain: float,
    analysis_seconds: float,
    decoder_threads: int,
    cache_path: Path | None,
    joint_motion_cache: dict[tuple[int, ...], np.ndarray],
) -> DelayResult:
    indices, camera_group = joint_indices_for_camera(video_key, joint_names)
    video_path = find_video_path(dataset_dir, video_key, episode_idx)
    analysis_frames = max(1, int(round(analysis_seconds * fps)))
    max_lag_frames = max(1, int(round(max_lag_seconds * fps)))
    # 차분 신호 N개에는 원본 프레임 N+1개가 필요하다.
    max_video_frames = analysis_frames + max_lag_frames + 1
    prefix_frames = min(max_video_frames, len(state))
    full_frames = len(state)
    prefix_cache_path = (
        cache_path.with_name(f"{cache_path.stem}_frames_{prefix_frames}.npz")
        if cache_path is not None else None
    )
    full_cache_path = (
        cache_path.with_name(f"{cache_path.stem}_frames_{full_frames}.npz")
        if cache_path is not None else None
    )
    index_key = tuple(int(index) for index in indices)
    if index_key not in joint_motion_cache:
        joint_motion_cache[index_key] = joint_motion_signal(state, indices)
    full_joint_motion = joint_motion_cache[index_key]
    global_joint_motion_std = float(
        np.std(np.linalg.norm(np.diff(state, axis=0), axis=1))
    )
    joint_motion = full_joint_motion[:analysis_frames]
    min_overlap_frames = max(15, int(round(0.60 * analysis_frames)))

    joint_std = float(np.std(joint_motion))
    early: dict[str, Any] = {}

    def early_needs_full(video_motion_prefix: np.ndarray) -> bool:
        video_std_prefix = float(np.std(video_motion_prefix))
        estimate_prefix: dict[str, Any] | None = None
        is_outlier_prefix = False
        status_prefix = "insufficient_motion"
        reason_prefix = "시작 구간의 영상 또는 관절 움직임이 부족함"
        # 상태 전체를 이미 읽었으므로 관절이 사실상 고정된 경우에는 영상을
        # 끝까지 디코딩해도 카메라 지연의 증거가 생길 수 없다.
        if joint_std < MIN_MEANINGFUL_JOINT_MOTION_STD:
            early.update({
                "estimate": None,
                "is_outlier": False,
                "status": (
                    "recording_error"
                    if (
                        global_joint_motion_std < MIN_MEANINGFUL_JOINT_MOTION_STD
                        and video_std_prefix >= 1e-3
                    )
                    else "insufficient_motion"
                ),
                "reason": (
                    "영상은 움직이지만 로봇 전체 관절 상태가 거의 고정됨"
                    if (
                        global_joint_motion_std < MIN_MEANINGFUL_JOINT_MOTION_STD
                        and video_std_prefix >= 1e-3
                    )
                    else "시작 구간의 대응 관절 움직임이 부족함"
                ),
                "video_std": video_std_prefix,
            })
            return False
        if (
            video_std_prefix >= 1e-3
            and joint_std >= MIN_MEANINGFUL_JOINT_MOTION_STD
        ):
            try:
                estimate_prefix = estimate_lag(
                    joint_motion, video_motion_prefix,
                    max_lag_frames=max_lag_frames,
                    smoothing_frames=smoothing_frames,
                    min_overlap_frames=min_overlap_frames,
                )
                is_outlier_prefix, status_prefix, reason_prefix = classify_delay(
                    estimate_prefix,
                    camera_group=camera_group,
                    fps=fps,
                    max_lag_frames=max_lag_frames,
                    min_delay_seconds=min_delay_seconds,
                    min_peak_correlation=min_peak_correlation,
                    min_correlation_gain=min_correlation_gain,
                    window_median=None,
                    window_mad=None,
                    reliable_windows=0,
                    video_motion_std=video_std_prefix,
                    joint_motion_std=joint_std,
                    global_joint_motion_std=global_joint_motion_std,
                )
            except ValueError:
                estimate_prefix = None
        early.update({
            "estimate": estimate_prefix,
            "is_outlier": is_outlier_prefix,
            "status": status_prefix,
            "reason": reason_prefix,
            "video_std": video_std_prefix,
        })
        return estimate_prefix is None or is_outlier_prefix

    video_motion, video_meta, cache_hit, full_verification = (
        load_or_decode_video_motion_staged(
            video_path,
            width=width,
            height=height,
            prefix_frames=prefix_frames,
            full_frames=full_frames,
            decoder_threads=decoder_threads,
            prefix_cache_path=prefix_cache_path,
            full_cache_path=full_cache_path,
            needs_full_verification=early_needs_full,
        )
    )

    if full_verification:
        joint_motion = full_joint_motion
        video_std = float(np.std(video_motion))
        joint_std = float(np.std(joint_motion))
        estimate = estimate_lag(
            joint_motion,
            video_motion,
            max_lag_frames=max_lag_frames,
            smoothing_frames=smoothing_frames,
            min_overlap_frames=max(30, int(round(3.0 * fps))),
        )
        is_outlier, status, reason = classify_delay(
            estimate,
            camera_group=camera_group,
            fps=fps,
            max_lag_frames=max_lag_frames,
            min_delay_seconds=min_delay_seconds,
            min_peak_correlation=min_peak_correlation,
            min_correlation_gain=min_correlation_gain,
            window_median=None,
            window_mad=None,
            reliable_windows=0,
            video_motion_std=video_std,
            joint_motion_std=joint_std,
            global_joint_motion_std=global_joint_motion_std,
        )
        reason = f"전체 영상 재검증; {reason}"
    else:
        # 캐시/신규 디코딩 모두 callback에서 동일한 초기 판정을 수행한다.
        estimate = early.get("estimate")
        is_outlier = bool(early.get("is_outlier", False))
        status = str(early.get("status", "insufficient_motion"))
        reason = str(early.get("reason", "시작 구간 판정 불가"))
        video_std = float(early.get("video_std", np.std(video_motion)))

    if estimate is None:
        # 짧은 에피소드 등에서 초기 구간 자체가 전체인 경우 한 번 더 시도한다.
        try:
            estimate = estimate_lag(
                joint_motion, video_motion,
                max_lag_frames=max_lag_frames,
                smoothing_frames=smoothing_frames,
                min_overlap_frames=min_overlap_frames,
            )
            is_outlier, status, reason = classify_delay(
                estimate,
                camera_group=camera_group,
                fps=fps,
                max_lag_frames=max_lag_frames,
                min_delay_seconds=min_delay_seconds,
                min_peak_correlation=min_peak_correlation,
                min_correlation_gain=min_correlation_gain,
                window_median=None,
                window_mad=None,
                reliable_windows=0,
                video_motion_std=video_std,
                joint_motion_std=joint_std,
                global_joint_motion_std=global_joint_motion_std,
            )
        except ValueError:
            pass

    estimate_available = estimate is not None
    if estimate is None:
        # 움직임 자체가 없어서 상관을 정의할 수 없는 것은 분석 실패가 아니다.
        # error 대신 판정 불가 상태와 null 진단값을 정상적으로 기록한다.
        is_outlier = False
        status = str(early.get("status", "insufficient_motion"))
        reason = str(early.get("reason", "유효한 시차 상관을 계산할 수 없음"))
    # 짧은 시작 판정 구간에서는 전체 상관이 가장 안정적이다. 이전의 겹치는 반구간
    # 재계산은 판정에 쓰이지 않으면서 같은 상관을 반복하므로 생략한다.
    window_median, window_mad, reliable_windows = None, None, 0
    reported_frame_count = int(video_meta["reported_frame_count"] or 0)
    frame_delta = reported_frame_count - len(state) if reported_frame_count else 0
    if (
        (reported_frame_count and frame_delta != 0)
        or video_meta["corrupt_count"]
        or video_meta["nonpositive_pts_steps"]
    ):
        reason += (
            f"; 영상 무결성 경고(frame_delta={frame_delta}, "
            f"corrupt={video_meta['corrupt_count']}, "
            f"nonpositive_pts={video_meta['nonpositive_pts_steps']})"
        )
    return DelayResult(
        episode_idx=episode_idx,
        camera=video_key,
        camera_group=camera_group,
        status=status,
        is_delay_outlier=is_outlier,
        delay_frames=int(estimate["delay_frames"]) if estimate_available else None,
        delay_seconds=(
            float(estimate["delay_frames"] / fps) if estimate_available else None
        ),
        peak_correlation=(
            _finite_float(estimate["peak_correlation"]) if estimate_available else None
        ),
        zero_lag_correlation=(
            _finite_float(estimate["zero_lag_correlation"])
            if estimate_available else None
        ),
        correlation_gain=(
            _finite_float(estimate["correlation_gain"]) if estimate_available else None
        ),
        peak_prominence=(
            _finite_float(estimate["peak_prominence"]) if estimate_available else None
        ),
        window_delay_median=window_median,
        window_delay_mad=window_mad,
        reliable_window_count=reliable_windows,
        video_frame_count=reported_frame_count,
        decoded_frame_count=int(video_meta["decoded_frame_count"]),
        parquet_row_count=len(state),
        frame_count_delta=frame_delta,
        video_duration_seconds=_finite_float(video_meta["reported_duration"]),
        analyzed_video_seconds=_finite_float(video_meta["analyzed_duration"]),
        pts_step_min=_finite_float(video_meta["pts_step_min"]),
        pts_step_max=_finite_float(video_meta["pts_step_max"]),
        nonpositive_pts_steps=int(video_meta["nonpositive_pts_steps"]),
        corrupt_frame_count=int(video_meta["corrupt_count"]),
        video_motion_std=video_std,
        joint_motion_std=joint_std,
        cache_hit=cache_hit,
        reason=reason,
    )


def analyze_episode_cameras(
    *,
    dataset_dir: Path,
    parquet_path: Path,
    video_keys: Sequence[str],
    joint_names: Sequence[str],
    fps: float,
    width: int,
    height: int,
    max_lag_seconds: float,
    min_delay_seconds: float,
    smoothing_frames: int,
    min_peak_correlation: float,
    min_correlation_gain: float,
    analysis_seconds: float,
    decoder_threads: int,
    cache_dir: Path | None,
) -> list[DelayResult]:
    try:
        state, episode_idx = load_state(parquet_path)
        state_error: Exception | None = None
    except Exception as error:
        state = np.empty((0, len(joint_names)))
        episode_idx = episode_index_from_path(parquet_path)
        state_error = error
    joint_motion_cache: dict[tuple[int, ...], np.ndarray] = {}
    episode_results: list[DelayResult] = []
    for video_key in video_keys:
        try:
            if state_error is not None:
                raise state_error
            cache_path = None
            if cache_dir is not None:
                safe_camera = video_key.replace("/", "_").replace(".", "_")
                cache_path = (
                    cache_dir / safe_camera / f"episode_{episode_idx:06d}.npz"
                )
            result = analyze_camera_episode(
                dataset_dir=dataset_dir,
                state=state,
                episode_idx=episode_idx,
                video_key=video_key,
                joint_names=joint_names,
                fps=fps,
                width=width,
                height=height,
                max_lag_seconds=max_lag_seconds,
                min_delay_seconds=min_delay_seconds,
                smoothing_frames=smoothing_frames,
                min_peak_correlation=min_peak_correlation,
                min_correlation_gain=min_correlation_gain,
                analysis_seconds=analysis_seconds,
                decoder_threads=decoder_threads,
                cache_path=cache_path,
                joint_motion_cache=joint_motion_cache,
            )
        except Exception as error:
            result = DelayResult(
                episode_idx=episode_idx, camera=video_key, camera_group="unknown",
                status="error", is_delay_outlier=False, delay_frames=None,
                delay_seconds=None, peak_correlation=None,
                zero_lag_correlation=None, correlation_gain=None,
                peak_prominence=None, window_delay_median=None,
                window_delay_mad=None, reliable_window_count=0,
                video_frame_count=0, decoded_frame_count=0,
                parquet_row_count=len(state), frame_count_delta=0,
                video_duration_seconds=None, analyzed_video_seconds=None,
                pts_step_min=None, pts_step_max=None,
                nonpositive_pts_steps=0, corrupt_frame_count=0,
                video_motion_std=None, joint_motion_std=None,
                cache_hit=False,
                reason=f"{type(error).__name__}: {error}",
            )
        episode_results.append(result)
    return episode_results


def analyze_dataset(
    dataset_dir: Path,
    *,
    cameras: Sequence[str] | None = None,
    episode_ids: set[int] | None = None,
    width: int = 96,
    height: int = 64,
    max_lag_seconds: float = 3.0,
    min_delay_seconds: float = 0.5,
    smoothing_frames: int = 5,
    min_peak_correlation: float = 0.55,
    min_correlation_gain: float = 0.08,
    analysis_seconds: float = 8.0,
    decoder_threads: int = 1,
    workers: int = 2,
    cache_dir: Path | None = None,
    verbose: bool = True,
) -> list[DelayResult]:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    info = load_dataset_info(dataset_dir)
    fps = float(info["fps"])
    state_info = info.get("features", {}).get("observation.state", {})
    joint_names = state_info.get("names")
    if not joint_names:
        raise ValueError("meta/info.json에 observation.state.names가 없습니다.")
    video_keys = discover_video_keys(info)
    if cameras:
        unknown = sorted(set(cameras) - set(video_keys))
        if unknown:
            raise ValueError(f"존재하지 않는 카메라: {unknown}; 사용 가능: {video_keys}")
        video_keys = list(cameras)
    parquet_files = discover_episode_files(dataset_dir)
    if episode_ids is not None:
        parquet_files = [
            path for path in parquet_files if episode_index_from_path(path) in episode_ids
        ]
        found = {episode_index_from_path(path) for path in parquet_files}
        if missing := episode_ids - found:
            raise ValueError(f"찾지 못한 에피소드: {sorted(missing)}")

    results: list[DelayResult] = []
    total = len(parquet_files) * len(video_keys)
    completed = 0
    common = {
        "dataset_dir": dataset_dir,
        "video_keys": video_keys,
        "joint_names": joint_names,
        "fps": fps,
        "width": width,
        "height": height,
        "max_lag_seconds": max_lag_seconds,
        "min_delay_seconds": min_delay_seconds,
        "smoothing_frames": smoothing_frames,
        "min_peak_correlation": min_peak_correlation,
        "min_correlation_gain": min_correlation_gain,
        "analysis_seconds": analysis_seconds,
        "decoder_threads": decoder_threads,
        "cache_dir": cache_dir,
    }
    if workers == 1:
        batches = (
            analyze_episode_cameras(parquet_path=path, **common)
            for path in parquet_files
        )
        for batch in batches:
            results.extend(batch)
            completed += len(batch)
            if verbose:
                print(f"[{completed:>4}/{total}] episode {batch[0].episode_idx:06d}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    analyze_episode_cameras, parquet_path=path, **common
                ): path
                for path in parquet_files
            }
            for future in as_completed(futures):
                batch = future.result()
                results.extend(batch)
                completed += len(batch)
                if verbose:
                    print(
                        f"[{completed:>4}/{total}] episode {batch[0].episode_idx:06d}",
                        flush=True,
                    )
    return results


def save_results(
    results: Sequence[DelayResult],
    *,
    dataset_dir: Path,
    output_dir: Path,
    parameters: dict[str, Any],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    csv_path = output_dir / "camera_delay_results.csv"
    json_path = output_dir / "camera_delay_summary.json"
    delete_path = output_dir / "camera_delay_delete_episodes.json"
    pd.DataFrame(rows).sort_values(["episode_idx", "camera"]).to_csv(csv_path, index=False)
    outliers = [row for row in rows if row["is_delay_outlier"]]
    errors = [row for row in rows if row["status"] == "error"]
    recording_errors = [row for row in rows if row["status"] == "recording_error"]
    insufficient = [row for row in rows if row["status"] == "insufficient_motion"]
    payload = {
        "schema_version": 3,
        "dataset": str(dataset_dir),
        "parameters": parameters,
        "analyzed_pairs": len(rows),
        "delay_outlier_count": len(outliers),
        "delay_outlier_episodes": sorted({row["episode_idx"] for row in outliers}),
        "delay_outliers": outliers,
        "recording_error_count": len(recording_errors),
        "recording_error_episodes": sorted({
            row["episode_idx"] for row in recording_errors
        }),
        "recording_errors": recording_errors,
        "insufficient_motion_count": len(insufficient),
        "error_count": len(errors),
        "errors": errors,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    delete_payload = {
        "schema_version": 1,
        "dataset": str(dataset_dir),
        "reason": "camera_delay",
        "delete_episode_count": len({row["episode_idx"] for row in outliers}),
        "delete_episode_indices": sorted({
            row["episode_idx"] for row in outliers
        }),
    }
    delete_path.write_text(
        json.dumps(delete_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return csv_path, json_path, delete_path


def parse_episode_spec(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    values: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"잘못된 에피소드 범위: {part}")
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", nargs="?", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--cameras", nargs="+",
        help="분석할 video key. 생략하면 meta/info.json의 모든 카메라를 분석합니다.",
    )
    parser.add_argument(
        "--episodes", help="예: 72-76 또는 1,3,8-10. 생략하면 전체 에피소드"
    )
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--max-lag-seconds", type=float, default=3.0)
    parser.add_argument(
        "--analysis-seconds", type=float, default=8.0,
        help=(
            "관절 움직임을 판정할 시작 구간(기본 8초: 문제 발생 5초 + 신뢰도 문맥 3초). "
            "영상은 여기에 max-lag만큼 더 읽습니다."
        ),
    )
    parser.add_argument("--min-delay-seconds", type=float, default=0.5)
    parser.add_argument("--smoothing-frames", type=int, default=5)
    parser.add_argument("--min-peak-correlation", type=float, default=0.55)
    parser.add_argument("--min-correlation-gain", type=float, default=0.08)
    parser.add_argument(
        "--workers", type=int, default=2,
        help="동시에 처리할 에피소드 수. 기본 2",
    )
    parser.add_argument("--decoder-threads", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.width < 8 or args.height < 8:
        raise ValueError("width와 height는 8 이상이어야 합니다.")
    if args.max_lag_seconds <= 0 or args.min_delay_seconds <= 0:
        raise ValueError("시차 관련 초 단위 옵션은 0보다 커야 합니다.")
    if args.analysis_seconds < 2.0:
        raise ValueError("analysis-seconds는 안정적인 상관 계산을 위해 2초 이상이어야 합니다.")
    if args.decoder_threads < 1:
        raise ValueError("decoder-threads는 1 이상이어야 합니다.")
    if args.workers < 1:
        raise ValueError("workers는 1 이상이어야 합니다.")
    if args.min_delay_seconds >= args.max_lag_seconds:
        raise ValueError("min-delay-seconds는 max-lag-seconds보다 작아야 합니다.")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else dataset_dir / "analysis_results"
    )
    cache_dir = None
    if not args.no_cache:
        cache_dir = (
            args.cache_dir.expanduser().resolve()
            if args.cache_dir else output_dir / "camera_delay_cache"
        )
    episode_ids = parse_episode_spec(args.episodes)
    parameters = {
        "cameras": args.cameras,
        "episodes": sorted(episode_ids) if episode_ids is not None else None,
        "width": args.width,
        "height": args.height,
        "max_lag_seconds": args.max_lag_seconds,
        "analysis_seconds": args.analysis_seconds,
        "min_delay_seconds": args.min_delay_seconds,
        "smoothing_frames": args.smoothing_frames,
        "min_peak_correlation": args.min_peak_correlation,
        "min_correlation_gain": args.min_correlation_gain,
        "decoder_threads": args.decoder_threads,
        "workers": args.workers,
        "cache_enabled": not args.no_cache,
        "cache_dir": str(cache_dir) if cache_dir else None,
    }
    results = analyze_dataset(
        dataset_dir,
        cameras=args.cameras,
        episode_ids=episode_ids,
        width=args.width,
        height=args.height,
        max_lag_seconds=args.max_lag_seconds,
        min_delay_seconds=args.min_delay_seconds,
        smoothing_frames=args.smoothing_frames,
        min_peak_correlation=args.min_peak_correlation,
        min_correlation_gain=args.min_correlation_gain,
        analysis_seconds=args.analysis_seconds,
        decoder_threads=args.decoder_threads,
        workers=args.workers,
        cache_dir=cache_dir,
        verbose=not args.quiet,
    )
    csv_path, json_path, delete_path = save_results(
        results, dataset_dir=dataset_dir, output_dir=output_dir, parameters=parameters
    )
    outliers = [result for result in results if result.is_delay_outlier]
    errors = [result for result in results if result.status == "error"]
    recording_errors = [
        result for result in results if result.status == "recording_error"
    ]
    print("\n카메라 지연 검출 완료")
    print(f"분석한 episode-camera 쌍: {len(results)}")
    print(f"지연 이상: {len(outliers)}개")
    for result in outliers:
        print(
            f"  - episode {result.episode_idx}: {result.camera} | "
            f"{result.delay_frames:+d} frames ({result.delay_seconds:+.3f}s) | "
            f"corr={result.peak_correlation:.3f}, gain={result.correlation_gain:.3f}"
        )
    if recording_errors:
        recording_episodes = sorted({
            result.episode_idx for result in recording_errors
        })
        print(f"녹화/상태 오류 의심 에피소드: {recording_episodes}")
    if errors:
        print(f"분석 오류: {len(errors)}개 (JSON의 errors 확인)")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"삭제 목록: {delete_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        elapsed = time.perf_counter() - PROGRAM_STARTED_AT
        print(f"총 실행 시간: {elapsed:.2f}초")
