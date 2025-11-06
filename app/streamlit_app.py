from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from effects import EffectEngine, EffectSettings, discover_assets

ASSETS_ROOT = Path(__file__).resolve().parent.parent / "assets"
BACKGROUND_DIR = ASSETS_ROOT / "backgrounds"
OVERLAY_DIR = ASSETS_ROOT / "overlays"

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def load_previews(paths: Dict[str, Path], size: int = 180) -> Dict[str, np.ndarray]:
    """Return resized previews for Streamlit display."""
    previews: Dict[str, np.ndarray] = {}
    for name, path in paths.items():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        target_h = max(1, int(size * image.shape[0] / image.shape[1]))
        preview = cv2.cvtColor(cv2.resize(image, (size, target_h)), cv2.COLOR_BGR2RGB)
        previews[name] = preview
    return previews


def create_cta():
    """Hero strip at the top of the page."""
    st.markdown(
        """
        <style>
            video {
                max-width: 520px;
                width: 100%;
                border-radius: 18px;
                box-shadow: 0 16px 45px rgba(10, 12, 24, 0.55);
                background-color: #05070F;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="padding: 2.5rem 2rem 1rem; background: linear-gradient(120deg,#1f2243,#121527 55%,rgba(124,108,244,0.3)); border-radius: 18px;">
            <h1 style="margin:0; font-size:2.4rem;">Realtime Vision Studio</h1>
            <p style="margin:0.6rem 0 0; font-size:1.1rem; color:rgba(246,246,251,0.85); max-width:560px;">
            Swap your background, layer cinematic lighting, and capture demo-ready footage in seconds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_asset_gallery(title: str, previews: Dict[str, np.ndarray]) -> None:
    if not previews:
        return
    st.markdown(f"#### {title}")
    cols = st.columns(3)
    for idx, (name, preview) in enumerate(previews.items()):
        with cols[idx % 3]:
            st.image(preview, caption=name, use_column_width=True)


class StudioProcessor(VideoProcessorBase):
    """Video processor applying the chosen segmentation effect in realtime."""

    def __init__(self, engine: EffectEngine, settings: EffectSettings) -> None:
        self._engine = engine
        self._settings = settings
        self._segmenter = self._create_segmenter()
        self._fps = 0.0
        self._last_timestamp = time.time()

    @staticmethod
    def _create_segmenter():
        import mediapipe as mp

        return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def update_settings(self, new_settings: EffectSettings) -> None:
        self._settings = new_settings

    def update_engine(self, engine: EffectEngine) -> None:
        self._engine = engine

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = self._segmenter.process(img_rgb)
        mask = result.segmentation_mask
        mask = (mask > self._settings.threshold).astype(np.float32)
        stylized = self._engine.render(img_bgr, mask, self._settings)
        self._track_fps()
        return av.VideoFrame.from_ndarray(stylized, format="bgr24")

    def _track_fps(self) -> None:
        now = time.time()
        elapsed = now - self._last_timestamp
        if elapsed > 0:
            self._fps = 1.0 / elapsed
        self._last_timestamp = now

    @property
    def fps(self) -> float:
        return self._fps


def init_session_state(defaults: EffectSettings) -> None:
    if "effect_settings" not in st.session_state:
        st.session_state.effect_settings = defaults


def build_sidebar(
    backgrounds: Dict[str, Path],
    overlays: Dict[str, Path],
) -> EffectSettings:
    settings: EffectSettings = st.session_state.effect_settings

    st.sidebar.markdown("### Control Panel")
    selected_effect = st.sidebar.selectbox(
        "Scene Atmosphere",
        (
            "Clean Cut",
            "Blurred Background",
            "Virtual Stage",
            "Aurora Overlay",
            "Duotone Portrait",
            "Sketch Silhouette",
            "Painterly Backdrop",
        ),
        index=0 if settings.effect_name not in (
            "Clean Cut",
            "Blurred Background",
            "Virtual Stage",
            "Aurora Overlay",
            "Duotone Portrait",
            "Sketch Silhouette",
            "Painterly Backdrop",
        ) else [
            "Clean Cut",
            "Blurred Background",
            "Virtual Stage",
            "Aurora Overlay",
            "Duotone Portrait",
            "Sketch Silhouette",
            "Painterly Backdrop",
        ].index(settings.effect_name),
    )

    threshold = st.sidebar.slider("Segmentation Threshold", 0.05, 0.75, settings.threshold, step=0.05)
    smooth = st.sidebar.slider("Edge Smoothing", 3, 31, settings.mask_smooth, step=2)

    updated = replace(settings, effect_name=selected_effect, threshold=threshold, mask_smooth=smooth)

    if updated.effect_name in {"Blurred Background", "Virtual Stage"}:
        blur = st.sidebar.slider("Background Blur", 5, 55, settings.blur_strength, step=2)
        updated = replace(updated, blur_strength=blur)

    if updated.effect_name == "Virtual Stage":
        background_names = ["None"] + list(backgrounds.keys())
        chosen_bg = st.sidebar.selectbox(
            "Virtual Background",
            background_names,
            index=background_names.index(settings.background_name) if settings.background_name in background_names else 0,
        )
        updated = replace(updated, background_name=None if chosen_bg == "None" else chosen_bg)

    if updated.effect_name == "Aurora Overlay":
        overlay_names = ["None"] + list(overlays.keys())
        chosen_overlay = st.sidebar.selectbox(
            "Light Overlay",
            overlay_names,
            index=overlay_names.index(settings.overlay_name) if settings.overlay_name in overlay_names else 0,
        )
        opacity = st.sidebar.slider("Overlay Intensity", 0.05, 0.75, settings.overlay_opacity, step=0.05)
        updated = replace(updated, overlay_name=None if chosen_overlay == "None" else chosen_overlay, overlay_opacity=opacity)

    if updated.effect_name == "Duotone Portrait":
        st.sidebar.caption("Applies soft pink and teal tones for a dreamy portrait look.")

    if updated.effect_name == "Painterly Backdrop":
        strength = st.sidebar.slider("Painterly Backdrop", 0.1, 1.0, settings.stylization_strength, step=0.1)
        updated = replace(updated, stylization_strength=strength)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Pro Tip:** Balance your room lighting, then record the demo with **cmd+shift+5** (macOS) or **Win+Alt+R**.
        """
    )

    st.session_state.effect_settings = updated
    return updated


def render_storyboard():
    with st.expander("Storyboard & Presentation Ideas", expanded=True):
        st.markdown(
            """
            - **Opening hook:** Start with `Clean Cut` to frame the subject crisply.
            - **Backdrop parade:** Cycle through Neon and Pastel scenes inside `Virtual Stage`.
            - **Mood switch:** Dial the `Aurora Overlay` slider to showcase lighting transitions.
            - **Creative finale:** Close with `Painterly Backdrop` layered with `Duotone Portrait`.
            """
        )


def main() -> None:
    st.set_page_config(
        page_title="Realtime Vision Studio",
        layout="wide",
        page_icon="🎨",
        initial_sidebar_state="expanded",
    )
    create_cta()

    backgrounds = discover_assets(BACKGROUND_DIR)
    overlays = discover_assets(OVERLAY_DIR)
    background_previews = load_previews(backgrounds)
    overlay_previews = load_previews(overlays)

    init_session_state(EffectSettings())
    current_settings = build_sidebar(backgrounds, overlays)
    render_storyboard()

    engine = EffectEngine(backgrounds, overlays)

    st.markdown("### Live Stage")
    stage_col, info_col = st.columns([1.4, 1])

    with stage_col:
        status_placeholder = st.empty()
        ctx = webrtc_streamer(
            key="vision-studio",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: StudioProcessor(engine=engine, settings=current_settings),
        )

    fps_placeholder = info_col.empty()
    info_col.markdown(
        """
        **Recording Tips**
        - Kick off with a clean framing to set the tone.
        - Rotate through a couple of backgrounds for visual variety.
        - Use the storyboard flow below to script your demo voiceover.
        """,
    )

    if ctx.video_processor:
        ctx.video_processor.update_engine(engine)
        ctx.video_processor.update_settings(current_settings)
        status_placeholder.success(
            f"🎥 Effect: {current_settings.effect_name} • Threshold: {current_settings.threshold:.2f}"
        )
        fps_placeholder.metric("Approx. FPS", f"{ctx.video_processor.fps:.1f}")
    else:
        status_placeholder.warning('Connect your camera and press "Start" to activate the studio.')

    st.markdown("---")
    cols = st.columns(2)
    with cols[0]:
        display_asset_gallery("Virtual Backgrounds", background_previews)
    with cols[1]:
        display_asset_gallery("Light Overlays", overlay_previews)

    st.markdown("### Demo Playbook")
    st.write(
        """
        - **A/B comparisons**: Record quick cuts of the same scene with different effects and edit side by side.
        - **Technical callouts**: Overlay small captions while adjusting the threshold to narrate what is happening.
        - **CTA moment**: End with a link or QR code pointing viewers to the project repo.
        """
    )


if __name__ == "__main__":
    main()
