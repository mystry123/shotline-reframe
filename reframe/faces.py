"""Face detection with YuNet (MIT licence, OpenCV model zoo).

Runs through OpenCV's DNN module, on CPU by default. Returns boxes in
*normalized picture-area coordinates* (0..1), so everything downstream is
independent of the analysis resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .decode import Frames

MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2026may.onnx"


@dataclass
class FaceDet:
    frame: int
    time: float
    x: float      # box left, normalized
    y: float      # box top
    w: float
    h: float
    score: float
    landmarks: np.ndarray  # (5, 2) normalized: right eye, left eye, nose, mouth right, mouth left
    estimated: bool = False  # a head estimated from the body (the face itself wasn't found); see persons.py

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def eye_y(self) -> float:
        return float(self.landmarks[:2, 1].mean())


class FaceDetector:
    def __init__(self, score_threshold: float = 0.6, nms_threshold: float = 0.3):
        self.detector = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), score_threshold, nms_threshold, 5000)

    def detect(self, frames: Frames) -> list[list[FaceDet]]:
        self.detector.setInputSize((frames.width, frames.height))
        out: list[list[FaceDet]] = []
        for i, frame in enumerate(frames.data):
            _, faces = self.detector.detect(np.ascontiguousarray(frame))
            dets: list[FaceDet] = []
            if faces is not None:
                for f in faces:
                    x, y, w, h = f[:4]
                    lm = f[4:14].reshape(5, 2) / np.array([frames.width, frames.height])
                    dets.append(FaceDet(
                        frame=i, time=i / frames.fps,
                        x=float(x / frames.width), y=float(y / frames.height),
                        w=float(w / frames.width), h=float(h / frames.height),
                        score=float(f[14]), landmarks=lm.astype(np.float32),
                    ))
            out.append(dets)
        return out
