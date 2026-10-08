"""Quality and framing settings.

Two kinds of setting, from two places:

- Plan settings (set per plan / per user by an admin in Shotline):
    quality        how closely the video is analysed: frames per second and
                   resolution of analysis. Higher finds smaller faces and
                   places cuts more precisely, and costs more CPU time.
    output_height  the plan's maximum output resolution (720, 1080, 1440, 2160).

- User choice per reframe:
    zoom           how far the engine may zoom in on small subjects:
                   "sharp"    never magnify the source much (default for low-res uploads)
                   "balanced" moderate zoom
                   "tight"    always allow up to 2× zoom, even if softer
"""

from __future__ import annotations

from dataclasses import dataclass

QUALITY = {
    #            analysis fps, analysis width (px), min upscale width for small sources
    "standard": {"fps": 5.0, "width": 640},
    "high":     {"fps": 8.0, "width": 960},
    "max":      {"fps": 12.0, "width": 1280},
}

# Largest allowed magnification of the source picture relative to the output.
ZOOM = {
    "sharp":    {"max_upscale": 1.5, "min_crop": 0.35},
    "balanced": {"max_upscale": 2.5, "min_crop": 0.35},
    "tight":    {"max_upscale": None, "min_crop": 0.5},   # always allow up to 2x
}

OUTPUT_HEIGHTS = (720, 1080, 1440, 2160)


@dataclass(frozen=True)
class Options:
    quality: str = "standard"
    output_height: int = 1080
    zoom: str = "balanced"

    def __post_init__(self) -> None:
        if self.quality not in QUALITY:
            raise ValueError(f"quality must be one of {sorted(QUALITY)}")
        if self.zoom not in ZOOM:
            raise ValueError(f"zoom must be one of {sorted(ZOOM)}")
        if self.output_height not in OUTPUT_HEIGHTS:
            raise ValueError(f"output_height must be one of {OUTPUT_HEIGHTS}")

    def min_crop_height(self, picture_px_height: int) -> float:
        """Smallest crop height (share of the picture) the camera may use."""
        z = ZOOM[self.zoom]
        if z["max_upscale"] is None:
            return z["min_crop"]
        by_resolution = self.output_height / (picture_px_height * z["max_upscale"])
        return float(min(1.0, max(z["min_crop"], by_resolution)))
