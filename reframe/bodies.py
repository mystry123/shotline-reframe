"""Person (body) detection with YOLOX-S (Apache-2.0, OpenCV model zoo).

Used to confirm that a face belongs to a person (ornaments, lamps and
patterns that a face detector fires on have no body), and as the framing
fallback when a face is turned away or too small. Preprocessing follows the
model zoo's reference demo exactly: RGB, letterboxed to 640×640 with grey
(114) padding, no normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .decode import Frames

MODEL = Path(__file__).resolve().parent.parent / "models" / "object_detection_yolox_2022nov.onnx"
INPUT = 640
STRIDES = (8, 16, 32)
PERSON = 0  # COCO class index


@dataclass
class BodyDet:
    frame: int
    time: float
    x: float   # normalized picture-area coordinates
    y: float
    w: float
    h: float
    score: float


class BodyDetector:
    def __init__(self, score_threshold: float = 0.35, nms_threshold: float = 0.5):
        self.net = cv2.dnn.readNet(str(MODEL))
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        grids, strides = [], []
        for s in STRIDES:
            n = INPUT // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def _detect_one(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        h, w = frame_bgr.shape[:2]
        ratio = min(INPUT / h, INPUT / w)
        padded = np.full((INPUT, INPUT, 3), 114.0, np.float32)
        resized = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        self.net.setInput(np.transpose(padded, (2, 0, 1))[None])
        out = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0]
        xy = (out[:, :2] + self.grids) * self.strides
        wh = np.exp(out[:, 2:4]) * self.strides
        scores = out[:, 4] * out[:, 5 + PERSON]
        keep = scores >= self.score_threshold
        if not keep.any():
            return []
        boxes = np.concatenate([xy[keep] - wh[keep] / 2, wh[keep]], axis=1) / ratio
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores[keep].tolist(), self.score_threshold, self.nms_threshold)
        return [(*boxes[i], float(scores[keep][i])) for i in np.array(idx).flatten()]

    def detect(self, frames: Frames, every: int = 1) -> list[list[BodyDet]]:
        """Bodies for every `every`-th frame (others empty)."""
        out: list[list[BodyDet]] = []
        for i, frame in enumerate(frames.data):
            if i % every:
                out.append([])
                continue
            dets = []
            for x, y, w, h, s in self._detect_one(frame):
                dets.append(BodyDet(i, i / frames.fps, float(x / frames.width), float(y / frames.height),
                                    float(w / frames.width), float(h / frames.height), s))
            out.append(dets)
        return out
