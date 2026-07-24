"""범용 analyze_outliers 결과를 실무 검수용 다중 페이지 PDF로 시각화."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from execution_logging import start_automatic_logging

if __name__ == "__main__":
    start_automatic_logging()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler

from analyze_outliers import DEFAULT_DATASET_DIR, analyze_dataset


PROGRAM_STARTED_AT = time.perf_counter()


FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
COLORS = {
    "navy": "#16324F", "blue": "#2878B5", "sky": "#8ECAE6",
    "orange": "#F4A261", "red": "#D62828", "green": "#2A9D8F",
    "gray": "#A7ADB4", "light": "#EEF2F5", "dark": "#20262E",
    "purple": "#7B2CBF",
}
MODE_COLORS = {
    "right_dominant": COLORS["blue"],
    "balanced": COLORS["green"],
    "left_dominant": COLORS["purple"],
    "all": COLORS["gray"],
}
SIGNAL_LABELS = {
    "length": "Length IQR",
    "joint_range": "Joint range",
    "motion_isolation": "Isolation Forest",
    "mode_cluster": "HDBSCAN + local density",
    "trajectory_dtw": "Mode-aware DTW",
    "annotation_integrity": "Annotation integrity",
    "annotation_structure": "Annotation structure",
    "no_motion": "No motion",
    "severe_truncation": "Severe truncation",
    "impact": "Impact (3×3-IQR)",
}


def configure_style() -> None:
    if Path(FONT_PATH).exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": "#B8BEC5",
        "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "grid.color": "#DDE2E7", "grid.alpha": 0.7,
        "pdf.fonttype": 3,
    })


def load_reviews(dataset_dir: Path) -> dict[int, dict]:
    path = dataset_dir / "meta" / "manual_review_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["episode_idx"]): row for row in payload.get("reviews", [])}


def load_camera_delay_deletes(dataset_dir: Path) -> set[int]:
    path = dataset_dir / "analysis_results" / "camera_delay_delete_episodes.json"
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(idx) for idx in payload.get("delete_episode_indices", [])}


def add_page_header(fig: plt.Figure, page: int, title: str, subtitle: str = "") -> None:
    fig.text(0.035, 0.955, f"{page:02d}", color="white", fontsize=16, weight="bold",
             ha="center", va="center",
             bbox={"boxstyle": "square,pad=0.55", "facecolor": "black", "edgecolor": "black"})
    fig.text(0.075, 0.958, title, fontsize=18, weight="bold", color=COLORS["dark"], va="center")
    if subtitle:
        fig.text(0.075, 0.928, subtitle, fontsize=9, color="#626A73", va="center")
    fig.add_artist(Line2D([0.075, 0.975], [0.91, 0.91], transform=fig.transFigure,
                          color="#AEB4BA", linewidth=0.8))
    fig.text(0.975, 0.025, str(page), ha="right", fontsize=8, color="#6C737A")


def save_page(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def annotate_ids(ax, x, y, ids, selected, color=COLORS["red"]) -> None:
    selected = set(selected)
    for px, py, episode in zip(x, y, ids):
        if int(episode) in selected:
            ax.annotate(str(int(episode)), (px, py), xytext=(4, 4),
                        textcoords="offset points", fontsize=7,
                        color=color, weight="bold")


def final_color(d: dict, episode: int) -> str:
    if episode in d["delete_labels"]:
        return COLORS["red"]
    if episode in d["validate_labels"]:
        return COLORS["orange"]
    return COLORS["green"]


def page_overview(
    pdf: PdfPages, d: dict, reviews: dict[int, dict], camera_deletes: set[int]
) -> None:
    fig = plt.figure(figsize=(13.33, 7.5))
    profile = d["complexity_profile"]
    subtitle = (
        f"{d['DATASET_DIR'].name} · {d['episode_count']} episodes · "
        f"{profile['level'].upper()} task · mode-aware quality analysis"
    )
    add_page_header(fig, 1, "LeRobot 데이터 품질 검증 요약", subtitle)
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.975, bottom=0.075, top=0.86,
                          width_ratios=[1.0, 1.45, 1.45], hspace=0.38, wspace=0.3)

    ax = fig.add_subplot(gs[:, 0]); ax.axis("off")
    reviewed_final = [idx for idx in d["all_outliers"] if idx in reviews]
    review_matches = sum(
        reviews[idx].get("status") == ("delete" if idx in d["delete_labels"] else "needs_review")
        for idx in reviewed_final
    )
    cards = [
        ("전체 / 분석 가능", f"{d['episode_count']} / {d['valid_episode_count']}", COLORS["navy"]),
        ("1차 CAMERA DELETE", len(camera_deletes), COLORS["red"]),
        ("DELETE", len(d["delete_labels"]), COLORS["red"]),
        ("VALIDATE", len(d["validate_labels"]), COLORS["orange"]),
        ("수동검수 일치", f"{review_matches}/{len(reviewed_final)}", COLORS["green"]),
    ]
    for row, (label, value, color) in enumerate(cards):
        y = 0.92 - row * 0.18
        ax.add_patch(plt.Rectangle((0.03, y - 0.11), 0.94, 0.14,
                                   facecolor="#F7F9FA", edgecolor="#D7DCE1"))
        ax.text(0.10, y - 0.01, str(value), fontsize=21, weight="bold", color=color, va="center")
        ax.text(0.10, y - 0.075, label, fontsize=8, color="#59616A", va="center")

    ax = fig.add_subplot(gs[0, 1:])
    names = list(d["signal_sets"])
    values = [len(d["signal_sets"][name]) for name in names]
    labels = [SIGNAL_LABELS.get(name, name) for name in names]
    bars = ax.barh(labels[::-1], values[::-1], color=COLORS["blue"])
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_title("분석 신호별 검출 수 — 최종 라벨과 별도")
    ax.set_xlabel("Episodes"); ax.grid(axis="x")
    ax.set_xlim(0, max(values + [1]) * 1.15 + 0.5)

    ax = fig.add_subplot(gs[1, 1])
    mode_counts = profile["hand_mode_counts"]
    mode_names = list(mode_counts)
    bars = ax.bar(mode_names, [mode_counts[m] for m in mode_names],
                  color=[MODE_COLORS.get(m, COLORS["gray"]) for m in mode_names])
    ax.bar_label(bars, padding=3)
    ax.set_title("손 사용 모드")
    ax.tick_params(axis="x", rotation=15); ax.grid(axis="y")

    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    signature = profile.get("dominant_annotation_signature")
    lines = [
        f"Complexity score   {profile['score']}",
        f"Median subtasks    {profile['median_subtask_count']}",
        f"Median skills      {profile['median_skill_count']}",
        f"Dominant signature {signature}",
        f"Adaptive PCA target {d['analysis_parameters']['pca_variance_target']:.0%}",
        "",
        f"DELETE   {sorted(d['delete_labels'])}",
        f"VALIDATE {sorted(d['validate_labels'])}",
    ]
    ax.text(0.03, 0.92, "판정 프로파일", fontsize=12, weight="bold", color=COLORS["dark"])
    ax.text(0.03, 0.80, "\n".join(lines), fontsize=9, va="top", linespacing=1.55,
            family="monospace", color="#3E464F")
    save_page(pdf, fig)


def page_integrity(pdf: PdfPages, d: dict) -> None:
    fig = plt.figure(figsize=(13.33, 7.5))
    bounds = d["length_bounds"]
    add_page_header(fig, 2, "무결성·길이·무동작 검사",
                    "Direct DELETE signals: unreadable data, no motion, severe truncation")
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.975, bottom=0.09, top=0.86,
                          hspace=0.36, wspace=0.27)
    frame = d["df_summary"]; ids = frame["episode_idx"].to_numpy()
    final = d["delete_labels"] | d["validate_labels"]
    colors = [final_color(d, int(idx)) for idx in ids]

    ax = fig.add_subplot(gs[0, :])
    ax.scatter(ids, frame["length"], c=colors, s=28, alpha=0.8)
    ax.axhspan(bounds["lower"], bounds["upper"], color=COLORS["green"], alpha=0.08)
    for value, label, style in [
        (bounds["lower"], "1.5-IQR", "--"), (bounds["upper"], None, "--"),
        (bounds["severe_lower"], "3-IQR", ":"), (bounds["severe_upper"], None, ":"),
    ]:
        ax.axhline(value, color=COLORS["red"], linestyle=style, linewidth=1,
                   label=label)
    annotate_ids(ax, ids, frame["length"], ids, final)
    ax.set(title="에피소드 길이", xlabel="Episode index", ylabel="Frames")
    ax.legend(); ax.grid()

    ax = fig.add_subplot(gs[1, 0])
    path = frame["total_joint_path"].to_numpy()
    ax.scatter(ids, np.maximum(path, 1e-8), c=colors, s=28)
    threshold = max(1e-6, 0.05 * float(np.median(path)))
    ax.axhline(threshold, color=COLORS["red"], linestyle="--",
               label=f"No-motion path threshold={threshold:.3g}")
    ax.set_yscale("log")
    annotate_ids(ax, ids, np.maximum(path, 1e-8), ids, d["no_motion_outliers"])
    ax.set(title="Total joint path", xlabel="Episode index", ylabel="Σ |Δ joint| (log)")
    ax.legend(); ax.grid()

    ax = fig.add_subplot(gs[1, 1])
    moving = frame["moving_fraction"].to_numpy()
    ax.scatter(ids, moving, c=colors, s=28)
    ax.axhline(0.10, color=COLORS["red"], linestyle="--", label="No-motion threshold=10%")
    annotate_ids(ax, ids, moving, ids, d["no_motion_outliers"])
    ax.set(title="움직임이 감지된 프레임 비율", xlabel="Episode index", ylabel="Moving fraction",
           ylim=(-0.03, 1.05))
    ax.legend(); ax.grid()
    save_page(pdf, fig)


def page_behavior(pdf: PdfPages, d: dict) -> None:
    fig = plt.figure(figsize=(13.33, 7.5))
    add_page_header(fig, 3, "수행 모드와 태스크 구조",
                    "Hand-use mode · normalized temporal activity · annotation signature")
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.975, bottom=0.09, top=0.86,
                          hspace=0.38, wspace=0.27)
    frame = d["df_summary"]; ids = frame["episode_idx"].to_numpy()
    modes = frame["hand_mode"].to_numpy(); share = frame["left_activity_share"].to_numpy()

    ax = fig.add_subplot(gs[0, :])
    ax.axhspan(0, 0.35, color=COLORS["blue"], alpha=0.07)
    ax.axhspan(0.35, 0.65, color=COLORS["green"], alpha=0.07)
    ax.axhspan(0.65, 1, color=COLORS["purple"], alpha=0.07)
    for mode in sorted(set(modes)):
        mask = modes == mode
        ax.scatter(ids[mask], share[mask], s=30, color=MODE_COLORS.get(mode, COLORS["gray"]),
                   label=f"{mode} ({mask.sum()})")
    annotate_ids(ax, ids, share, ids, d["all_outliers"])
    ax.set(title="좌우 팔 활동 비율과 hand mode", xlabel="Episode index",
           ylabel="Left activity share", ylim=(-0.03, 1.03))
    ax.legend(ncol=3); ax.grid()

    ax = fig.add_subplot(gs[1, 0])
    temporal = d["temporal_activity_feature_matrix"].reshape(len(frame), 3, 4)
    x = np.arange(1, 5)
    group_names = ["Left arm", "Right arm", "Grippers"]
    group_colors = [COLORS["purple"], COLORS["blue"], COLORS["orange"]]
    for group_idx, (name, color) in enumerate(zip(group_names, group_colors)):
        mean = temporal[:, group_idx, :].mean(axis=0)
        std = temporal[:, group_idx, :].std(axis=0)
        ax.plot(x, mean, marker="o", color=color, label=name)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)
    ax.set(title="정규화된 시간 구간별 평균 활동", xlabel="Normalized time bin",
           ylabel="Within-group activity share", xticks=x)
    ax.legend(); ax.grid()

    ax = fig.add_subplot(gs[1, 1])
    available = frame[frame["annotation_available"]]
    if len(available):
        pairs = Counter(zip(available["subtask_count"].astype(int),
                            available["skill_count"].astype(int)))
        for (subtasks, skills), count in pairs.items():
            selected = any(
                int(row.episode_idx) in d["annotation_structure_outliers"]
                for row in available[(available["subtask_count"] == subtasks)
                                     & (available["skill_count"] == skills)].itertuples()
            )
            ax.scatter(subtasks, skills, s=45 + count * 7,
                       color=COLORS["red"] if selected else COLORS["sky"],
                       edgecolor="white")
            ax.annotate(f"{count} eps", (subtasks, skills), xytext=(5, 3),
                        textcoords="offset points", fontsize=8)
        signature = d["complexity_profile"].get("dominant_annotation_signature")
        if signature:
            ax.scatter(signature[0], signature[1], s=240, facecolors="none",
                       edgecolors=COLORS["green"], linewidth=2, label="Dominant signature")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Annotation unavailable", ha="center", va="center",
                transform=ax.transAxes, color=COLORS["gray"])
    ax.set(title="Annotation 구조 분포", xlabel="Subtask count", ylabel="Skill count")
    ax.grid()
    save_page(pdf, fig)


def _analysis_feature_names(d: dict) -> list[str]:
    names = [
        f"{block}·{joint}"
        for block in d["motion_block_names"] for joint in d["task_joint_names"]
    ]
    names.extend(
        f"Temporal {group} bin{idx}"
        for group in ("left", "right", "gripper") for idx in range(1, 5)
    )
    return names


def page_isolation(pdf: PdfPages, d: dict) -> None:
    fig = plt.figure(figsize=(13.33, 7.5))
    add_page_header(fig, 4, "모드별 Isolation Forest",
                    "Robust motion + temporal features · threshold calculated inside each hand mode")
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.98, bottom=0.10, top=0.86,
                          height_ratios=[1, 1.08], hspace=0.42, wspace=0.28)
    frame = d["df_summary"]; ids = frame["episode_idx"].to_numpy()
    scores = frame["isolation_score"].to_numpy(); modes = frame["hand_mode"].to_numpy()
    finite = np.isfinite(scores); outliers = set(d["isolation_outliers"])

    ax = fig.add_subplot(gs[0, 0])
    for mode in sorted(set(modes)):
        mask = (modes == mode) & finite
        ax.scatter(ids[mask], scores[mask], s=28, color=MODE_COLORS.get(mode, COLORS["gray"]),
                   label=mode)
        threshold = d["isolation_thresholds"].get(str(mode))
        if threshold is not None and np.isfinite(threshold):
            ax.axhline(threshold, color=MODE_COLORS.get(mode, COLORS["gray"]),
                       linestyle="--", linewidth=1)
    mask = np.isin(ids, list(outliers)) & finite
    ax.scatter(ids[mask], scores[mask], s=70, marker="X", color=COLORS["red"], label="Signal")
    annotate_ids(ax, ids[mask], scores[mask], ids[mask], ids[mask])
    ax.set(title="Isolation score", xlabel="Episode index", ylabel="Score (lower = rarer)")
    ax.legend(ncol=2); ax.grid()

    ax = fig.add_subplot(gs[0, 1])
    for mode in sorted(set(modes)):
        values = scores[(modes == mode) & finite]
        if len(values):
            ax.hist(values, bins=14, alpha=0.45, color=MODE_COLORS.get(mode), label=mode)
    ax.set(title="모드별 score 분포", xlabel="Isolation score", ylabel="Count")
    ax.legend(); ax.grid(axis="y")

    ax = fig.add_subplot(gs[1, :])
    candidates = sorted(outliers | set(d["delete_labels"]) | set(d["validate_labels"]))
    rows = [np.flatnonzero(ids == idx)[0] for idx in candidates if idx in set(ids)]
    matrix = d["analysis_feature_matrix"]
    if rows:
        z = StandardScaler().fit_transform(matrix)
        top = np.argsort(np.mean(np.abs(z[rows]), axis=0))[-14:]
        heat = z[np.ix_(rows, top)]
        vmax = max(2.5, float(np.nanpercentile(np.abs(heat), 95)))
        image = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(len(rows)), [f"Ep {ids[row]}" for row in rows])
        feature_names = _analysis_feature_names(d)
        ax.set_xticks(range(len(top)), [feature_names[col] for col in top], rotation=32, ha="right")
        fig.colorbar(image, ax=ax, pad=0.01, label="Standardized feature")
    else:
        ax.text(0.5, 0.5, "No Isolation/final candidates", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("후보의 주요 motion/temporal feature")
    save_page(pdf, fig)


def page_cluster(pdf: PdfPages, d: dict) -> None:
    fig = plt.figure(figsize=(13.33, 7.5))
    diag = d["cluster_diagnostics"]
    add_page_header(fig, 5, "적응형 PCA + 모드별 HDBSCAN",
                    f"Global PCA: {diag.get('global_pca_components')} components · "
                    f"explained variance {diag.get('global_explained_variance', 0):.1%}")
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.975, bottom=0.09, top=0.86,
                          hspace=0.38, wspace=0.27)
    frame = d["df_summary"]; ids = frame["episode_idx"].to_numpy()
    embedding = d["cluster_embedding"]; modes = frame["hand_mode"].to_numpy()

    ax = fig.add_subplot(gs[:, 0])
    for mode in sorted(set(modes)):
        mask = modes == mode
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=28,
                   color=MODE_COLORS.get(mode, COLORS["gray"]), alpha=0.65, label=mode)
    candidate_mask = np.isin(ids, d["cluster_outliers"])
    ax.scatter(embedding[candidate_mask, 0], embedding[candidate_mask, 1],
               s=85, marker="X", color=COLORS["red"], label="Confirmed density signal")
    annotate_ids(ax, embedding[:, 0], embedding[:, 1], ids, d["all_outliers"])
    ax.set(title="Global adaptive PCA projection", xlabel="PC1", ylabel="PC2")
    ax.legend(); ax.grid()

    mode_names = [name for name in sorted(d["hand_mode_counts"]) if name in diag]
    raw = [diag[name].get("outlier_count", 0) for name in mode_names]
    confirmed = [diag[name].get("confirmed_outlier_count", 0) for name in mode_names]
    ax = fig.add_subplot(gs[0, 1]); x = np.arange(len(mode_names)); width = 0.36
    ax.bar(x - width/2, raw, width, color=COLORS["gray"], label="Raw HDBSCAN noise")
    ax.bar(x + width/2, confirmed, width, color=COLORS["red"], label="Confirmed local-density signal")
    ax.set_xticks(x, mode_names, rotation=15); ax.set_ylabel("Episodes")
    ax.set_title("HDBSCAN noise 보수화"); ax.legend(); ax.grid(axis="y")

    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    lines = []
    for name in mode_names:
        item = diag[name]
        if item.get("status") == "protected_small_mode":
            lines.append(f"{name}: n={item['size']} · protected small mode")
        else:
            lines.append(
                f"{name}: n={item['size']} · PCA={item.get('pca_components')} · "
                f"variance={item.get('explained_variance', 0):.1%} · "
                f"clusters={item.get('cluster_count')}"
            )
    ax.text(0.02, 0.92, "모드별 clustering diagnostics", fontsize=12, weight="bold")
    ax.text(0.02, 0.75, "\n\n".join(lines), fontsize=9, va="top", color="#3E464F")
    save_page(pdf, fig)


def page_dtw(pdf: PdfPages, d: dict) -> None:
    fig = plt.figure(figsize=(13.33, 7.5))
    add_page_header(fig, 6, "모드별 DTW 궤적 비교",
                    "Relative task-joint trajectory · robust scaling · nearest trajectories within hand mode")
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.975, bottom=0.09, top=0.86,
                          hspace=0.38, wspace=0.27)
    frame = d["df_summary"]; ids = frame["episode_idx"].to_numpy()
    scores = frame["dtw_score"].to_numpy(); modes = frame["hand_mode"].to_numpy()
    finite = np.isfinite(scores); mask = np.isin(ids, d["dtw_outliers"]) & finite

    ax = fig.add_subplot(gs[0, :])
    for mode in sorted(set(modes)):
        selected = (modes == mode) & finite
        ax.scatter(ids[selected], scores[selected], s=28,
                   color=MODE_COLORS.get(mode, COLORS["gray"]), label=mode)
        threshold = d["dtw_thresholds"].get(str(mode))
        if threshold is not None and np.isfinite(threshold):
            ax.axhline(threshold, color=MODE_COLORS.get(mode, COLORS["gray"]),
                       linestyle="--", linewidth=1)
    ax.scatter(ids[mask], scores[mask], s=75, marker="X", color=COLORS["red"], label="DTW signal")
    annotate_ids(ax, ids[mask], scores[mask], ids[mask], ids[mask])
    ax.set(title="모드 내부 nearest-trajectory DTW score", xlabel="Episode index", ylabel="DTW score")
    ax.legend(ncol=4); ax.grid()

    trajectories = np.asarray(d["normalized_trajectories"])
    rms = np.sqrt(np.mean(trajectories ** 2, axis=2))
    time = np.linspace(0, 1, rms.shape[1])
    ax = fig.add_subplot(gs[1, 0])
    normal = ~np.isin(ids, list(d["all_outliers"]))
    if np.any(normal):
        mean = rms[normal].mean(axis=0); std = rms[normal].std(axis=0)
        ax.fill_between(time, mean-std, mean+std, color=COLORS["sky"], alpha=0.3)
        ax.plot(time, mean, color=COLORS["navy"], linewidth=2, label="Pass mean ±1σ")
    ax.set(title="Pass trajectory envelope", xlabel="Normalized time", ylabel="RMS joint trajectory")
    ax.legend(); ax.grid()

    ax = fig.add_subplot(gs[1, 1])
    final_ids = sorted(d["all_outliers"])
    for episode in final_ids[:10]:
        row = np.flatnonzero(ids == episode)
        if len(row):
            ax.plot(time, rms[row[0]], linewidth=1.8, label=f"Ep {episode}",
                    color=final_color(d, episode))
    if not final_ids:
        ax.text(0.5, 0.5, "No final candidates", ha="center", va="center", transform=ax.transAxes)
    ax.set(title="최종 후보 trajectory", xlabel="Normalized time", ylabel="RMS joint trajectory")
    ax.legend(ncol=2); ax.grid()
    save_page(pdf, fig)


def page_decision(
    pdf: PdfPages, d: dict, reviews: dict[int, dict], camera_deletes: set[int]
) -> None:
    fig = plt.figure(figsize=(13.33, 7.5))
    profile = d["complexity_profile"]
    policy = "annotation structure priority" if profile["level"] == "complex" else "robust motion priority"
    add_page_header(fig, 7, "신호 교차판정과 최종 라벨", f"Decision policy: {policy}")
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.98, bottom=0.10, top=0.85,
                          width_ratios=[1.75, 1], wspace=0.31)
    signal_names = list(d["signal_sets"])
    raw_ids = set().union(*d["signal_sets"].values()) | set(d["all_outliers"])
    # 페이지 가독성을 위해 신호 수가 많은 순으로 최대 40개를 표시하되 최종 후보는 보존한다.
    ranked = sorted(raw_ids, key=lambda idx: (-d["evidence_counts"].get(int(idx), 0), int(idx)))
    shown = list(sorted(d["all_outliers"]))
    shown.extend(idx for idx in ranked if idx not in set(shown))
    candidate_ids = shown[:40]
    sets = [set(d["signal_sets"][name]) for name in signal_names]
    matrix = np.array([[int(idx in values) for values in sets] for idx in candidate_ids], dtype=int)

    ax = fig.add_subplot(gs[0, 0])
    if len(candidate_ids):
        ax.imshow(matrix, aspect="auto", cmap=ListedColormap(["#F0F2F4", COLORS["red"]]),
                  vmin=0, vmax=1)
        ax.set_xticks(range(len(signal_names)), [SIGNAL_LABELS.get(n, n) for n in signal_names],
                      rotation=30, ha="right")
        review_abbr = {"pass": "P", "needs_review": "R", "delete": "D"}
        ylabels = []
        for idx in candidate_ids:
            model = "D" if idx in d["delete_labels"] else "V" if idx in d["validate_labels"] else "–"
            manual = review_abbr.get(reviews.get(idx, {}).get("status"), "–")
            ylabels.append(f"Ep {idx}  [model={model}, manual={manual}]")
        ax.set_yticks(range(len(candidate_ids)), ylabels)
        for row in range(len(candidate_ids)):
            for col in range(len(signal_names)):
                if matrix[row, col]:
                    ax.text(col, row, "●", ha="center", va="center", color="white", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No signals", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("후보별 raw signal matrix")

    ax = fig.add_subplot(gs[0, 1])
    categories = ["CAMERA\nDELETE", "DELETE", "VALIDATE", "PASS"]
    values = [
        len(camera_deletes), len(d["delete_labels"]),
        len(d["validate_labels"]), len(d["pass_labels"]),
    ]
    bars = ax.bar(
        categories, values,
        color=[COLORS["navy"], COLORS["red"], COLORS["orange"], COLORS["green"]],
    )
    ax.bar_label(bars, padding=4)
    ax.set_title("최종 모델 라벨"); ax.set_ylabel("Episodes"); ax.grid(axis="y")
    final_ids = sorted(d["all_outliers"])
    reviewed = [idx for idx in final_ids if idx in reviews]
    review_lines = [
        f"Ep {idx}: model={'delete' if idx in d['delete_labels'] else 'validate'} / "
        f"manual={reviews[idx].get('status')}"
        for idx in reviewed[:12]
    ]
    camera_line = f"1차 camera delete: {sorted(camera_deletes)}"
    ax.text(0.02, -0.20, camera_line + "\n\n수동검수 대조\n"
            + ("\n".join(review_lines) or "No reviewed final candidates"),
            transform=ax.transAxes, fontsize=8, va="top", color="#3E464F")
    save_page(pdf, fig)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LeRobot 범용 품질 분석 PDF 생성")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-pdf", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    configure_style()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_pdf = (
        args.output_pdf.expanduser().resolve()
        if args.output_pdf else dataset_dir / "analysis_results" / "outlier_analysis_report.pdf"
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    analysis = analyze_dataset(dataset_dir, write_outputs=False, verbose=False)
    reviews = load_reviews(dataset_dir)
    camera_deletes = load_camera_delay_deletes(dataset_dir)
    with PdfPages(output_pdf, metadata={
        "Title": "LeRobot Generalized Data Quality Report",
        "Author": "ROBOTIS data quality analysis",
        "Subject": "Episode-level integrity, behavior mode, and anomaly diagnostics",
    }) as pdf:
        page_overview(pdf, analysis, reviews, camera_deletes)
        page_integrity(pdf, analysis)
        page_behavior(pdf, analysis)
        page_isolation(pdf, analysis)
        page_cluster(pdf, analysis)
        page_dtw(pdf, analysis)
        page_decision(pdf, analysis, reviews, camera_deletes)
    print(f"PDF 리포트 생성 완료: {output_pdf}")


if __name__ == "__main__":
    try:
        main()
    finally:
        elapsed = time.perf_counter() - PROGRAM_STARTED_AT
        print(f"총 실행 시간: {elapsed:.2f}초")
