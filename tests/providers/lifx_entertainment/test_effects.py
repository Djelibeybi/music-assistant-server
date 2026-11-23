"""Tests of the matrix channel grid, the pixel effects and the audio features they read."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from music_assistant.providers.lifx_entertainment import bridge as bridge_module
from music_assistant.providers.lifx_entertainment.areas import MATRIX_GRID, build_channels
from music_assistant.providers.lifx_entertainment.constants import COLOR_MODES, DEFAULT_COLOR_MODE
from music_assistant.providers.lifx_entertainment.effects import (
    HSBK_MAX,
    AudioFeatures,
    Effect,
    SoundLevel,
    Spectrum,
    Waveform,
    create_effect,
)
from music_assistant.providers.lifx_entertainment.features import FeatureTracker
from music_assistant.providers.lifx_entertainment.models import (
    EntertainmentArea,
    LifxDevice,
    ZoneSupport,
)
from music_assistant.providers.lifx_entertainment.session import EntertainmentSession
from music_assistant.providers.sendspin_lights.models import LightColor

_TILE_PRODUCT = 55
_STRIP_PRODUCT = 32

_LOUD = AudioFeatures(loudness=0.9, bands=[0.9, 0.7, 0.5, 0.3, 0.1], beat=0.0, centroid=0.2)


@dataclass
class _StubAnimator:
    """Records the frames a session sends to one light."""

    canvas_width: int
    canvas_height: int
    duration_ms: int = 0
    frames: list[list[tuple[int, int, int, int]]] = field(default_factory=list)

    @property
    def pixel_count(self) -> int:
        return self.canvas_width * self.canvas_height

    def send_frame(self, frame: list[tuple[int, int, int, int]]) -> Any:
        self.frames.append(frame)
        return SimpleNamespace(gated=False)


def _tile() -> LifxDevice:
    return LifxDevice("d073d5eeeeee", "127.0.0.1", label="Tile", product_id=_TILE_PRODUCT)


def _strip(zones: int = 16) -> LifxDevice:
    return LifxDevice(
        "d073d5bbbbbb",
        "127.0.0.1",
        label="Strip",
        product_id=_STRIP_PRODUCT,
        zone_support=ZoneSupport.EXTENDED,
        zone_count=zones,
    )


def _streaming_session(
    devices: list[LifxDevice], animators: dict[str, _StubAnimator], **kwargs: Any
) -> EntertainmentSession:
    """Return a session that streams to stub animators, without any network I/O."""
    area = EntertainmentArea("all", "All", devices, build_channels(devices))
    session = EntertainmentSession(MagicMock(), MagicMock(), area, **kwargs)
    session._animators.update(animators)  # type: ignore[arg-type]
    session._streaming = True
    return session


def test_matrix_is_split_into_a_grid_of_channels() -> None:
    """A matrix gets one channel per grid cell, columns along x and rows along z."""
    columns, rows = MATRIX_GRID
    channels = build_channels([_tile()])
    assert len(channels) == columns * rows
    assert {(c.column, c.row) for c in channels} == {
        (x, y) for x in range(columns) for y in range(rows)
    }
    by_cell = {(c.column, c.row): c for c in channels}
    # columns run left to right, the top row sits highest
    assert by_cell[(0, 0)].position[0] < by_cell[(columns - 1, 0)].position[0]
    assert by_cell[(0, 0)].position[2] > by_cell[(0, rows - 1)].position[2]
    assert by_cell[(0, 0)].position[0] == by_cell[(0, rows - 1)].position[0]


def test_matrix_sits_beside_other_lights_on_the_x_axis() -> None:
    """A matrix next to a strip takes its own columns of the x axis."""
    channels = build_channels([_strip(), _tile()])
    strip_x = {c.position[0] for c in channels if c.service_id == "d073d5bbbbbb"}
    tile_x = {c.position[0] for c in channels if c.service_id == "d073d5eeeeee"}
    assert len(tile_x) == MATRIX_GRID[0]
    assert max(strip_x) < min(tile_x)


def test_analyzer_colours_paint_matrix_cells() -> None:
    """Each channel's colour covers its block of the matrix's pixels."""
    tile = _tile()
    animator = _StubAnimator(8, 8)
    session = _streaming_session([tile], {tile.serial: animator})
    cell = {(c.column, c.row): c.channel_id for c in session.area.channels}
    session.send(
        [
            LightColor(cell[(0, 0)], red=65535),
            LightColor(cell[(2, 2)], blue=65535),
        ]
    )
    frame = animator.frames[-1]
    assert len(frame) == 64
    top_left, bottom_right, middle = frame[0], frame[63], frame[4 * 8 + 4]
    assert top_left[2] > 0
    assert bottom_right[2] > 0
    assert top_left[0] != bottom_right[0]
    # cells that got no colour stay dark
    assert middle[2] == 0


def test_create_effect_falls_back_to_sound_level_on_a_bulb() -> None:
    """A single pixel can't show a waveform or bars, so it shows the sound level."""
    for effect in Effect:
        assert isinstance(create_effect(effect, 1, 1), SoundLevel)
    assert isinstance(create_effect(Effect.WAVEFORM, 16, 1), Waveform)
    assert isinstance(create_effect(Effect.SPECTRUM, 8, 8), Spectrum)


def test_sound_level_follows_the_loudness() -> None:
    """Louder audio renders brighter white; a bass-heavy spectrum renders warmer."""
    quiet, loud = SoundLevel(), SoundLevel()
    quiet_frame = loud_frame = []
    for t in range(10):
        quiet_frame = quiet.render(t / 20, AudioFeatures(loudness=0.05), 4, 1)
        loud_frame = loud.render(t / 20, AudioFeatures(loudness=0.95, centroid=0.0), 4, 1)
    assert len(set(loud_frame)) == 1
    assert loud_frame[0][1] == 0
    assert loud_frame[0][2] > quiet_frame[0][2]
    assert loud_frame[0][3] < quiet_frame[0][3]


def test_waveform_spreads_the_spectrum_across_the_strip() -> None:
    """Bass sits at the start of the strip in red, treble at the end in blue."""
    effect = Waveform()
    frame = []
    for t in range(20):
        frame = effect.render(t / 20, _LOUD, 16, 1)
    assert len(frame) == 16
    assert frame[0][0] < frame[-1][0]
    # the bass-heavy test spectrum lights the start brighter than the end
    assert frame[0][2] > frame[-1][2]


def test_waveform_repeats_its_row_down_a_matrix() -> None:
    """A matrix shows the same columns in every row."""
    frame = Waveform().render(0.0, _LOUD, 8, 4)
    assert len(frame) == 32
    assert all(frame[y * 8 : (y + 1) * 8] == frame[:8] for y in range(4))


def test_spectrum_draws_bars_from_the_bottom_up() -> None:
    """On a matrix, a loud band fills its column from the bottom row."""
    width, height = 8, 8
    frame = Spectrum().render(0.0, _LOUD, width, height)
    bottom_left = frame[(height - 1) * width]
    top_right = frame[width - 1]
    assert bottom_left[2] > 0
    assert top_right[2] == 0


def test_spectrum_shows_bar_brightness_along_a_strip() -> None:
    """On a strip, each zone's brightness is its band's energy."""
    frame = Spectrum().render(0.0, _LOUD, 16, 1)
    assert len(frame) == 16
    assert frame[0][2] > frame[-1][2] > 0


def test_feature_tracker_waits_for_the_playhead() -> None:
    """Frames apply once the playhead reaches them; onsets count as a beat once."""
    tracker = FeatureTracker()
    tracker.push(_frame(1_000, loudness=65535, peak=200, spectrum=[0, 65535]))
    tracker.push(_frame(2_000, loudness=0, peak=None, spectrum=[65535, 0]))

    assert tracker.advance(500).loudness == 0.0
    first = tracker.advance(1_500)
    assert first.loudness == 1.0
    assert first.beat == 1.0
    assert first.centroid == 1.0
    second = tracker.advance(2_500)
    assert second.loudness == 0.0
    assert second.beat == 0.0
    assert second.centroid == 0.0

    tracker.clear()
    assert tracker.advance(3_000) == AudioFeatures()


def test_feature_tracker_ignores_weak_onsets() -> None:
    """An onset below the strength cut is not a beat."""
    tracker = FeatureTracker()
    tracker.push(_frame(1_000, peak=50))
    assert tracker.advance(1_000).beat == 0.0


def test_effect_modes_are_colour_modes() -> None:
    """The effects are extra choices of the existing mode setting."""
    assert COLOR_MODES[0] == DEFAULT_COLOR_MODE
    assert set(COLOR_MODES) >= {e.value for e in Effect}
    assert bridge_module._effect_for("waveform") == Effect.WAVEFORM
    assert bridge_module._effect_for("smooth") is None
    assert bridge_module._analyzer_mode("spectrum") == DEFAULT_COLOR_MODE
    assert bridge_module._analyzer_mode("energetic") == "energetic"


@pytest.mark.parametrize("brightness", [100, 50, 0])
def test_session_renders_effects_on_every_light(brightness: int) -> None:
    """A strip and a matrix render the effect; a bulb shows the sound level."""
    bulb = LifxDevice("d073d5aaaaaa", "127.0.0.1", label="Bulb", product_id=27)
    strip, tile = _strip(), _tile()
    animators = {
        bulb.serial: _StubAnimator(1, 1),
        strip.serial: _StubAnimator(16, 1),
        tile.serial: _StubAnimator(8, 8),
    }
    session = _streaming_session(
        [bulb, strip, tile], animators, effect=Effect.WAVEFORM, brightness=brightness
    )
    session.send_effect(_LOUD)

    bulb_frame = animators[bulb.serial].frames[-1]
    strip_frame = animators[strip.serial].frames[-1]
    tile_frame = animators[tile.serial].frames[-1]
    assert len(bulb_frame) == 1
    assert bulb_frame[0][1] == 0
    assert len(strip_frame) == 16
    assert len(tile_frame) == 64
    if brightness == 0:
        assert all(p[2] == 0 for p in strip_frame + tile_frame + bulb_frame)
    else:
        assert len({p[0] for p in strip_frame}) > 1
        assert max(p[2] for p in strip_frame) <= HSBK_MAX * brightness // 100


def test_session_switches_effects_while_streaming() -> None:
    """Changing the effect takes hold on the next frame; None stops the effect frames."""
    strip = _strip()
    animator = _StubAnimator(16, 1)
    session = _streaming_session([strip], {strip.serial: animator}, effect=Effect.SOUND_LEVEL)
    session.send_effect(_LOUD)
    assert len({p[0] for p in animator.frames[-1]}) == 1

    session.effect = Effect.WAVEFORM
    session.send_effect(_LOUD)
    assert len({p[0] for p in animator.frames[-1]}) > 1

    sent = len(animator.frames)
    session.effect = None
    session.send_effect(_LOUD)
    assert len(animator.frames) == sent


def _frame(
    timestamp_us: int,
    *,
    loudness: int | None = None,
    peak: int | None = None,
    spectrum: list[int] | None = None,
) -> Any:
    return SimpleNamespace(
        timestamp_us=timestamp_us,
        loudness=loudness,
        peak=peak,
        spectrum=None if spectrum is None else np.array(spectrum, dtype=np.uint16),
    )
