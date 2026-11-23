"""
Audio-reactive pixel effects for LIFX strips and matrices.

Adapted from the ``soundlevel``, ``waveform`` and ``spectrum2d`` effects of Glowup
(https://github.com/pkivolowitz/glowup) by Perry Kivolowitz. The effects read their audio
features from the Sendspin visualizer instead of Glowup's signal bus, render onto a
``width x height`` canvas (a strip is one row, a bulb a single pixel), and take Glowup's
default parameters as constants.
"""

# MIT License
#
# Copyright (c) 2026 Perry Kivolowitz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

HSBK_MAX = 65535
KELVIN_DEFAULT = 3500
KELVIN_MIN = 1500
KELVIN_MAX = 9000

Pixel = tuple[int, int, int, int]


class Effect(StrEnum):
    """The effects, by the colour mode that selects them."""

    SOUND_LEVEL = "sound_level"
    WAVEFORM = "waveform"
    SPECTRUM = "spectrum"


@dataclass
class AudioFeatures:
    """The audio at the playhead, each value in [0, 1]."""

    # Loudness on the visualizer's dB scale (-60 dB is 0, full scale is 1).
    loudness: float = 0.0
    # Spectrum bands from bass to treble, on the same dB scale.
    bands: list[float] = field(default_factory=list)
    # 1.0 on the frame an onset lands, else 0.0.
    beat: float = 0.0
    # Where the spectrum's energy sits: 0 bass-heavy, 1 treble-heavy.
    centroid: float = 0.5


class AudioEffect(ABC):
    """An effect that renders one frame per call, keeping its own smoothing state."""

    @abstractmethod
    def render(self, t: float, audio: AudioFeatures, width: int, height: int) -> list[Pixel]:
        """
        Render one frame, row-major with row 0 at the top.

        :param t: Seconds since the effect started.
        :param audio: The audio at the playhead.
        :param width: Canvas width in pixels.
        :param height: Canvas height in pixels.
        """


def create_effect(effect: Effect, width: int, height: int) -> AudioEffect:
    """
    Create an effect for a canvas, using sound level where the effect needs more pixels.

    :param effect: The chosen effect.
    :param width: Canvas width in pixels.
    :param height: Canvas height in pixels.
    """
    if width * height <= 1:
        return SoundLevel()
    if effect == Effect.WAVEFORM:
        return Waveform()
    if effect == Effect.SPECTRUM:
        return Spectrum()
    return SoundLevel()


class SoundLevel(AudioEffect):
    """Brightness follows the loudness and flashes on beats; colour temperature follows the tone."""

    BASE_BRIGHTNESS = 0.30
    MAX_BRIGHTNESS = 1.0
    MIN_BRIGHTNESS = 0.05
    SENSITIVITY = 1.0
    BEAT_FLASH = 0.20
    BEAT_FLASH_DECAY_S = 0.3
    SMOOTH_ALPHA = 0.4
    WARM_KELVIN = 2700
    COOL_KELVIN = 5000

    def __init__(self) -> None:
        """Initialize the smoothing state."""
        self._smooth_bri = self.BASE_BRIGHTNESS
        self._flash = _BeatFlash(self.BEAT_FLASH_DECAY_S)

    def render(self, t: float, audio: AudioFeatures, width: int, height: int) -> list[Pixel]:
        """Render every pixel in the same white."""
        level = min(1.0, audio.loudness * self.SENSITIVITY)
        target = self.BASE_BRIGHTNESS + level * (self.MAX_BRIGHTNESS - self.BASE_BRIGHTNESS)
        self._smooth_bri += self.SMOOTH_ALPHA * (target - self._smooth_bri)
        brightness = min(
            1.0, self._smooth_bri + self._flash.update(t, audio.beat) * self.BEAT_FLASH
        )
        brightness = max(self.MIN_BRIGHTNESS, brightness)
        kelvin = int(self.WARM_KELVIN + audio.centroid * (self.COOL_KELVIN - self.WARM_KELVIN))
        kelvin = max(KELVIN_MIN, min(KELVIN_MAX, kelvin))
        return [(0, 0, int(brightness * HSBK_MAX), kelvin)] * (width * height)


class Waveform(AudioEffect):
    """Each column shows one part of the spectrum, bass to treble, hue from red to blue."""

    BASS_HUE = 0.0
    TREBLE_HUE = 240.0
    MIN_BRIGHTNESS = 0.05
    MAX_BRIGHTNESS = 1.0
    SENSITIVITY = 1.5
    SATURATION = HSBK_MAX
    BEAT_FLASH = 0.30
    BEAT_FLASH_DECAY_S = 0.2
    CONTRAST = 1.0
    NOISE_GATE = 0.15
    SMOOTH_ALPHA = 0.35

    def __init__(self) -> None:
        """Initialize the smoothing state."""
        self._smooth: list[float] = []
        self._flash = _BeatFlash(self.BEAT_FLASH_DECAY_S)

    def render(self, t: float, audio: AudioFeatures, width: int, height: int) -> list[Pixel]:
        """Render one row of columns and repeat it down a matrix."""
        if len(self._smooth) != width:
            self._smooth = [0.0] * width
        flash = self._flash.update(t, audio.beat) * self.BEAT_FLASH
        bands = audio.bands or [0.0] * 8
        row: list[Pixel] = []
        for x in range(width):
            energy = _band_energy(bands, x, width)
            if energy < self.NOISE_GATE:
                energy = 0.0
            energy = min(1.0, energy * self.SENSITIVITY)
            if self.CONTRAST != 1.0 and energy > 0.0:
                energy **= self.CONTRAST
            self._smooth[x] += self.SMOOTH_ALPHA * (energy - self._smooth[x])
            brightness = self.MIN_BRIGHTNESS + self._smooth[x] * (
                self.MAX_BRIGHTNESS - self.MIN_BRIGHTNESS
            )
            brightness = min(1.0, brightness + flash)
            fraction = x / max(1, width - 1)
            hue = (self.BASS_HUE + fraction * (self.TREBLE_HUE - self.BASS_HUE)) % 360.0
            row.append(
                (_hue_to_u16(hue), self.SATURATION, int(brightness * HSBK_MAX), KELVIN_DEFAULT)
            )
        return row * height


class Spectrum(AudioEffect):
    """Vertical spectrum bars with a peak-hold marker; a strip shows the bars as brightness."""

    PEAK_DECAY = 0.85
    PEAK_THRESHOLD = 0.05
    HUE_RANGE = 0.66
    BAR_BRI_MIN = 0.25
    BAR_BRI_MAX = 0.76
    PEAK_BRI = 0.90
    PEAK_SAT = 0.50
    BRIGHTNESS = 0.76

    def __init__(self) -> None:
        """Initialize the peak-hold state."""
        self._peaks: list[float] = []

    def render(self, t: float, audio: AudioFeatures, width: int, height: int) -> list[Pixel]:
        """Render the bars from the bottom row up."""
        bands = audio.bands or [0.0] * 8
        energies = [_band_energy(bands, x, width) for x in range(width)]
        if len(self._peaks) != width:
            self._peaks = [0.0] * width
        for x, energy in enumerate(energies):
            self._peaks[x] = max(energy, self._peaks[x] * self.PEAK_DECAY)
        beat_mult = min(1.5, 1.0 + audio.beat * 0.5)
        if height == 1:
            # A strip has no height for bars, so each zone's brightness is its bar.
            return [
                (
                    self._hue(x, width),
                    HSBK_MAX,
                    min(HSBK_MAX, int(HSBK_MAX * energies[x] * self.BRIGHTNESS * beat_mult)),
                    KELVIN_DEFAULT,
                )
                for x in range(width)
            ]
        pixels: list[Pixel] = []
        for y in range(height):
            from_bottom = height - 1 - y
            for x in range(width):
                pixels.append(self._pixel(x, width, height, from_bottom, energies[x], beat_mult))
        return pixels

    def _pixel(
        self, x: int, width: int, height: int, from_bottom: int, energy: float, beat_mult: float
    ) -> Pixel:
        hue = self._hue(x, width)
        bar_height = energy * height
        peak_row = int(self._peaks[x] * (height - 1))
        if from_bottom == peak_row and self._peaks[x] > self.PEAK_THRESHOLD:
            return (
                hue,
                int(HSBK_MAX * self.PEAK_SAT),
                int(HSBK_MAX * self.PEAK_BRI * self.BRIGHTNESS),
                KELVIN_DEFAULT,
            )
        if from_bottom < bar_height:
            # Brighter toward the top of the bar.
            fill = from_bottom / max(1.0, bar_height)
            fraction = self.BAR_BRI_MIN + (self.BAR_BRI_MAX - self.BAR_BRI_MIN) * fill
            brightness = min(HSBK_MAX, int(HSBK_MAX * fraction * self.BRIGHTNESS * beat_mult))
            return (hue, HSBK_MAX, brightness, KELVIN_DEFAULT)
        return (0, 0, 0, KELVIN_DEFAULT)

    def _hue(self, x: int, width: int) -> int:
        return int((x / max(1, width - 1)) * self.HUE_RANGE * HSBK_MAX)


class _BeatFlash:
    """A flash that starts on a beat and fades out linearly."""

    def __init__(self, decay_s: float) -> None:
        self._decay_s = decay_s
        self._level = 0.0
        self._started = 0.0

    def update(self, t: float, beat: float) -> float:
        """Return the flash level in [0, 1] at time ``t``."""
        if beat > 0.5 and self._level < 0.1:
            self._level = 1.0
            self._started = t
        elif self._level > 0.0:
            self._level = max(0.0, 1.0 - (t - self._started) / self._decay_s)
        return self._level


def _band_energy(bands: list[float], x: int, width: int) -> float:
    """Interpolate the band energy for column ``x`` of ``width``."""
    count = len(bands)
    position = x * (count - 1) / max(1, width - 1)
    low = int(position)
    high = min(low + 1, count - 1)
    fraction = position - low
    return bands[low] * (1.0 - fraction) + bands[high] * fraction


def _hue_to_u16(degrees: float) -> int:
    return int(degrees / 360.0 * HSBK_MAX) % (HSBK_MAX + 1)
