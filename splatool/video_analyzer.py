"""ゲームプレイ動画からフレームを抽出し、注目シーン候補を検出するモジュール。

Switchのキャプチャ(30秒クリップ)やキャプチャボード録画のMP4を想定。
- 一定間隔でフレームをサンプリング
- 隣接フレーム間のヒストグラム差分から「画面が大きく変わった瞬間」
  (キル・デス・スペシャル発動・リスポーンなど)を注目シーン候補として検出
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Frame:
    """抽出した1フレーム"""

    timestamp: float  # 動画内の秒数
    jpeg_bytes: bytes
    is_key_moment: bool = False  # 注目シーン候補かどうか
    change_score: float = 0.0  # 直前フレームからの変化量 (0-1)

    @property
    def b64(self) -> str:
        return base64.standard_b64encode(self.jpeg_bytes).decode("utf-8")

    @property
    def label(self) -> str:
        m, s = divmod(int(self.timestamp), 60)
        return f"{m}:{s:02d}"


@dataclass
class VideoAnalysis:
    """動画1本ぶんの抽出結果"""

    duration: float
    fps: float
    total_frames: int
    frames: list[Frame] = field(default_factory=list)

    @property
    def key_moments(self) -> list[Frame]:
        return [f for f in self.frames if f.is_key_moment]


def _encode_jpeg(frame: np.ndarray, max_width: int = 1280, quality: int = 80) -> bytes:
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEGエンコードに失敗しました")
    return buf.tobytes()


def _hist_diff(a: np.ndarray, b: np.ndarray) -> float:
    """HSVヒストグラムの相関から変化量(0=同一, 1=完全に別画面)を返す"""
    score = 0.0
    for img_a, img_b in ((a, b),):
        hsv_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
        for ch in range(2):  # H, S チャンネル(明滅の影響が大きいVは除外)
            ha = cv2.calcHist([hsv_a], [ch], None, [32], [0, 256])
            hb = cv2.calcHist([hsv_b], [ch], None, [32], [0, 256])
            cv2.normalize(ha, ha)
            cv2.normalize(hb, hb)
            corr = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
            score += (1.0 - max(corr, 0.0)) / 2
    return min(score, 1.0)


def extract_frames(
    video_path: str,
    interval_sec: float = 3.0,
    max_frames: int = 40,
    min_change: float = 0.15,
) -> VideoAnalysis:
    """動画からフレームを抽出し、変化量の大きい瞬間に注目シーンフラグを立てる。

    注目シーンは固定しきい値ではなく、動画全体の変化量分布に対して
    「中央値 + 1σ 以上」かつ min_change 以上のフレームを相対判定する。
    通常のプレー中もカメラ移動で画面は変わり続けるため、その動画の中で
    特に変化が大きい瞬間(キル・デス・スペシャル等)だけを拾う狙い。

    Args:
        video_path: 動画ファイルのパス
        interval_sec: サンプリング間隔(秒)
        max_frames: 抽出上限。超える場合は間隔を自動で広げる
        min_change: 注目シーン判定の最低変化量(0-1)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"動画を開けませんでした: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps else 0.0

        # 上限を超えないよう間隔を調整
        if duration > 0 and duration / interval_sec > max_frames:
            interval_sec = duration / max_frames

        analysis = VideoAnalysis(duration=duration, fps=fps, total_frames=total)

        prev_small: np.ndarray | None = None
        t = 0.0
        while t < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                break

            small = cv2.resize(frame, (320, 180))
            change = _hist_diff(prev_small, small) if prev_small is not None else 0.0
            prev_small = small

            analysis.frames.append(
                Frame(
                    timestamp=t,
                    jpeg_bytes=_encode_jpeg(frame),
                    change_score=change,
                )
            )
            t += interval_sec

        # 相対判定: 変化量が「中央値 + 1σ」を超え、かつ最低値以上のフレームを注目シーンに
        scores = np.array([f.change_score for f in analysis.frames[1:]])
        if len(scores) >= 3:
            threshold = max(float(np.median(scores) + scores.std()), min_change)
        else:
            threshold = min_change
        for f in analysis.frames:
            f.is_key_moment = f.change_score >= threshold

        return analysis
    finally:
        cap.release()


def select_frames_for_ai(analysis: VideoAnalysis, max_send: int = 20) -> list[Frame]:
    """AIに送るフレームを選ぶ。注目シーンを優先し、残りは等間隔で埋める。"""
    frames = analysis.frames
    if len(frames) <= max_send:
        return frames

    selected: dict[float, Frame] = {}
    for f in sorted(analysis.key_moments, key=lambda x: -x.change_score)[: max_send // 2]:
        selected[f.timestamp] = f

    remaining = max_send - len(selected)
    if remaining > 0:
        step = max(len(frames) // remaining, 1)
        for f in frames[::step]:
            if len(selected) >= max_send:
                break
            selected.setdefault(f.timestamp, f)

    return sorted(selected.values(), key=lambda x: x.timestamp)
