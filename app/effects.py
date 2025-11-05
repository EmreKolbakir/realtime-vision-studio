"""
Utilities for applying visual effects to video frames using segmentation masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

ASSET_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".ppm")


def _load_image(path: Path) -> np.ndarray:
    """Read an image file using OpenCV. Raises if the image cannot be loaded."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot load image at {path}")
    return image


def discover_assets(folder: Path) -> Dict[str, Path]:
    """
    Return a dict mapping user-friendly names to image paths discovered in a folder.
    Nested folders are ignored to keep curation simple.
    """
    if not folder.exists():
        return {}
    assets: Dict[str, Path] = {}
    for file in folder.iterdir():
        if file.suffix.lower() in ASSET_EXTENSIONS and file.is_file():
            pretty_name = file.stem.replace("_", " ").title()
            assets[pretty_name] = file
    return dict(sorted(assets.items()))


def resize_to_frame(image: np.ndarray, frame_shape: Tuple[int, int, int]) -> np.ndarray:
    """Resize an image so it fits the current frame while keeping aspect ratio."""
    frame_h, frame_w = frame_shape[:2]
    img_h, img_w = image.shape[:2]
    scale = max(frame_w / img_w, frame_h / img_h)
    resized = cv2.resize(image, (int(img_w * scale), int(img_h * scale)))
    # Crop the center if necessary
    start_x = max(0, (resized.shape[1] - frame_w) // 2)
    start_y = max(0, (resized.shape[0] - frame_h) // 2)
    return resized[start_y : start_y + frame_h, start_x : start_x + frame_w]


def smooth_mask(mask: np.ndarray, blur_kernel: int) -> np.ndarray:
    """Smooth the mask to reduce jitter between frames."""
    mask32 = mask.astype(np.float32)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    return cv2.GaussianBlur(mask32, (blur_kernel, blur_kernel), 0)


def apply_background_swap(
    frame: np.ndarray, mask: np.ndarray, background: np.ndarray
) -> np.ndarray:
    """Replace the background of the frame with the provided image."""
    background_resized = resize_to_frame(background, frame.shape)
    mask_3c = np.dstack([mask] * 3)
    return np.uint8(frame * mask_3c + background_resized * (1.0 - mask_3c))


def apply_blur_background(frame: np.ndarray, mask: np.ndarray, intensity: int) -> np.ndarray:
    """Blur the background while keeping the subject sharp."""
    intensity = max(3, intensity | 1)  # gkernel must be odd
    blurred = cv2.GaussianBlur(frame, (intensity, intensity), 0)
    mask_3c = np.dstack([mask] * 3)
    return np.uint8(frame * mask_3c + blurred * (1.0 - mask_3c))


def apply_overlay(frame: np.ndarray, overlay: np.ndarray, opacity: float) -> np.ndarray:
    """Blend an overlay texture on top of the full frame."""
    overlay_resized = resize_to_frame(overlay, frame.shape)
    return cv2.addWeighted(frame, 1.0 - opacity, overlay_resized, opacity, 0)


def apply_duotone(frame: np.ndarray, mask: np.ndarray, colors: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> np.ndarray:
    """Apply a duotone treatment to the foreground."""
    fg_color, bg_color = colors
    normalized = frame.astype(np.float32) / 255.0
    luminance = np.dot(normalized[..., :3], [0.299, 0.587, 0.114])
    fg = np.zeros_like(frame, dtype=np.float32)
    for channel, color_component in enumerate(fg_color):
        fg[..., channel] = luminance * (color_component / 255.0)
    bg = np.zeros_like(frame, dtype=np.float32)
    for channel, color_component in enumerate(bg_color):
        bg[..., channel] = luminance * (color_component / 255.0)
    mask_3c = np.dstack([mask] * 3)
    mix = fg * mask_3c + bg * (1.0 - mask_3c)
    return np.uint8(np.clip(mix * 255.0, 0, 255))


def apply_sketch(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Generate a sketch outline around the silhouette."""
    mask_3c = np.dstack([mask] * 3).astype(np.float32)
    edges = cv2.Canny(frame, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR).astype(np.float32)
    silhouette = 1.0 - mask_3c
    outline = cv2.dilate(silhouette, np.ones((3, 3), np.float32), iterations=1)
    glow = cv2.GaussianBlur(outline, (11, 11), 0)
    glow = np.clip(glow * 220.0, 0, 255.0)
    combined = frame.astype(np.float32) * mask_3c + edges_colored * silhouette + glow
    return np.uint8(np.clip(combined, 0, 255))


def apply_stylization(frame: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """Apply OpenCV's stylization filter to the background for a painterly look."""
    try:
        stylized = cv2.stylization(frame, sigma_s=60, sigma_r=0.6)
    except cv2.error:
        stylized = cv2.medianBlur(frame, 11)
    mask_3c = np.dstack([mask] * 3)
    return np.uint8(frame * mask_3c + cv2.addWeighted(frame, 1.0 - strength, stylized, strength, 0) * (1.0 - mask_3c))


@dataclass
class EffectSettings:
    effect_name: str = "Clean Cut"
    threshold: float = 0.35
    mask_smooth: int = 7
    blur_strength: int = 25
    overlay_name: str | None = None
    overlay_opacity: float = 0.25
    background_name: str | None = None
    stylization_strength: float = 0.6
    duotone_palette: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
        (255, 115, 168),
        (120, 220, 255),
    )

    def as_dict(self) -> Dict[str, str]:
        return {
            "effect": self.effect_name,
            "background": self.background_name or "None",
            "overlay": self.overlay_name or "None",
        }


class EffectEngine:
    """Encapsulate effect logic and provide an easy API for the video processor."""

    def __init__(
        self,
        backgrounds: Dict[str, Path],
        overlays: Dict[str, Path],
    ) -> None:
        self.backgrounds = backgrounds
        self.overlays = overlays
        self._bg_cache: Dict[str, np.ndarray] = {}
        self._overlay_cache: Dict[str, np.ndarray] = {}

    def _resolve_background(self, name: str | None, frame_shape: Tuple[int, int, int]) -> np.ndarray | None:
        if not name or name not in self.backgrounds:
            return None
        if name not in self._bg_cache:
            self._bg_cache[name] = _load_image(self.backgrounds[name])
        return resize_to_frame(self._bg_cache[name], frame_shape)

    def _resolve_overlay(self, name: str | None, frame_shape: Tuple[int, int, int]) -> np.ndarray | None:
        if not name or name not in self.overlays:
            return None
        if name not in self._overlay_cache:
            self._overlay_cache[name] = _load_image(self.overlays[name])
        return resize_to_frame(self._overlay_cache[name], frame_shape)

    def render(self, frame: np.ndarray, mask: np.ndarray, settings: EffectSettings) -> np.ndarray:
        mask = smooth_mask(mask, settings.mask_smooth)
        mask = np.clip(mask, 0.0, 1.0)
        effect = settings.effect_name

        if effect == "Clean Cut":
            return frame

        if effect == "Blurred Background":
            return apply_blur_background(frame, mask, settings.blur_strength)

        if effect == "Virtual Stage":
            background = self._resolve_background(settings.background_name, frame.shape)
            if background is not None:
                return apply_background_swap(frame, mask, background)
            return apply_blur_background(frame, mask, settings.blur_strength)

        if effect == "Aurora Overlay":
            overlay = self._resolve_overlay(settings.overlay_name, frame.shape)
            if overlay is not None:
                combined = apply_overlay(frame, overlay, settings.overlay_opacity)
            else:
                combined = apply_overlay(frame, np.full(frame.shape, 180, dtype=np.uint8), settings.overlay_opacity)
            return np.uint8(combined)

        if effect == "Duotone Portrait":
            return apply_duotone(frame, mask, settings.duotone_palette)

        if effect == "Sketch Silhouette":
            return apply_sketch(frame, mask)

        if effect == "Painterly Backdrop":
            return apply_stylization(frame, mask, settings.stylization_strength)

        return frame
