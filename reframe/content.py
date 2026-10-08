"""Where the content is, for shots with no people (graphics, slides, app
promos, screen recordings, products).

Per analysed frame: the box holding what stands out from the background
(text, UI, objects), measured as local contrast against a heavily blurred copy
of the frame plus edges. Bounds are the union of all clearly visible regions
(specks and soft gradients are ignored).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .decode import Frames


@dataclass
class ContentBox:
    time: float
    x0: float   # normalized picture-area coordinates
    y0: float
    x1: float
    y1: float
    mass: float  # share of the frame that is content (0 = blank)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def content_boxes(frames: Frames) -> list[ContentBox | None]:
    out: list[ContentBox | None] = []
    for i, frame in enumerate(frames.data):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(h, w) / 12)
        contrast = np.abs(gray - background)
        edges = cv2.Canny(gray.astype(np.uint8), 60, 160).astype(np.float32)
        saliency = cv2.GaussianBlur(contrast / (contrast.max() + 1e-6) + edges / 255.0, (0, 0), 3)
        mask = saliency > max(0.12, float(np.percentile(saliency, 75)))
        mass = float(mask.mean())
        if mass < 0.002:
            out.append(None)  # blank frame (fade, flash, empty background)
            continue
        # Union of every clearly visible region (a thin line of text next to a
        # bold logo still counts); specks below 0.05% of the frame are noise.
        joined = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        min_area = 0.0005 * h * w
        keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= min_area]
        if not keep:
            out.append(None)
            continue
        x0 = min(stats[k, cv2.CC_STAT_LEFT] for k in keep) / w
        y0 = min(stats[k, cv2.CC_STAT_TOP] for k in keep) / h
        x1 = max(stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH] for k in keep) / w
        y1 = max(stats[k, cv2.CC_STAT_TOP] + stats[k, cv2.CC_STAT_HEIGHT] for k in keep) / h
        out.append(ContentBox(i / frames.fps, float(x0), float(y0), float(x1), float(y1), mass))
    return out
