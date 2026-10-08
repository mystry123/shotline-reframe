"""Text inside the picture: burned-in captions, titles, slide text, UI text.

PP-OCRv3 text detection (Apache-2.0, OpenCV model zoo), run about once a
second. Used to:
- keep captions and titles in frame when following a person;
- recognise screens (channel pages, screen recordings, slides full of small
  text), where faces in thumbnails are not subjects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .decode import Frames

MODEL = Path(__file__).resolve().parent.parent / "models" / "text_detection_en_ppocrv3_2023may.onnx"


@dataclass
class TextBox:
    frame: int
    time: float
    x0: float   # normalized picture-area coordinates
    y0: float
    x1: float
    y1: float

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


class TextDetector:
    def __init__(self) -> None:
        self.model = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(str(MODEL)))
        self.model.setBinaryThreshold(0.3)
        self.model.setPolygonThreshold(0.5)
        self.model.setUnclipRatio(2.0)
        self.model.setMaxCandidates(200)

    def detect(self, frames: Frames, every: int = 1) -> list[list[TextBox]]:
        w = max(32, frames.width - frames.width % 32)
        h = max(32, frames.height - frames.height % 32)
        self.model.setInputParams(1.0 / 255.0, (w, h), (123.675, 116.28, 103.53), True)
        scale = np.array([1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225])
        out: list[list[TextBox]] = []
        for i, frame in enumerate(frames.data):
            if i % every:
                out.append([])
                continue
            img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            self.model.setInputScale(1.0 / 255.0 * scale)
            polys, _ = self.model.detect(img)
            boxes = []
            for p in polys:
                p = np.asarray(p, np.float32)
                x0, y0 = p.min(axis=0)
                x1, y1 = p.max(axis=0)
                boxes.append(TextBox(i, i / frames.fps, float(x0 / w), float(y0 / h), float(x1 / w), float(y1 / h)))
            out.append(boxes)
        return out
