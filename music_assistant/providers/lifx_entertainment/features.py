"""Turn the Sendspin visualizer frames into the audio features the effects read."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from .effects import AudioFeatures

if TYPE_CHECKING:
    from aiosendspin.server.roles.visualizer.features import ExtractedFrame

_U16_MAX = 65535
# Onsets weaker than this (0-255) don't count as a beat, the same cut WLED Audio Sync uses.
PEAK_MIN_STRENGTH = 100
# Frames arrive at most a few seconds ahead of the playhead; this bounds the queue
# should the renderer stall.
_MAX_PENDING = 512


class FeatureTracker:
    """Hold visualizer frames until the playhead reaches them, then report the features."""

    def __init__(self) -> None:
        """Initialize an empty tracker."""
        self._pending: deque[ExtractedFrame] = deque(maxlen=_MAX_PENDING)
        self._loudness = 0.0
        self._bands: list[float] = []

    def push(self, frame: ExtractedFrame) -> None:
        """
        Queue a frame; frames arrive ahead of the playhead, in timestamp order.

        :param frame: The visualizer frame.
        """
        self._pending.append(frame)

    def clear(self) -> None:
        """Drop the queued frames and the current features, as after a seek or a stop."""
        self._pending.clear()
        self._loudness = 0.0
        self._bands = []

    def advance(self, now_us: int) -> AudioFeatures:
        """
        Apply the frames due at ``now_us`` and return the features there.

        :param now_us: The playhead on the Sendspin server clock.
        """
        beat = 0.0
        while self._pending and self._pending[0].timestamp_us <= now_us:
            frame = self._pending.popleft()
            if frame.loudness is not None:
                self._loudness = frame.loudness / _U16_MAX
            if frame.spectrum is not None:
                self._bands = [float(v) / _U16_MAX for v in frame.spectrum.tolist()]
            if frame.peak is not None and frame.peak >= PEAK_MIN_STRENGTH:
                beat = 1.0
        return AudioFeatures(
            loudness=self._loudness,
            bands=list(self._bands),
            beat=beat,
            centroid=_centroid(self._bands),
        )


def _centroid(bands: list[float]) -> float:
    """Return where the spectrum's energy sits, 0 at the lowest band and 1 at the highest."""
    total = sum(bands)
    if len(bands) < 2 or total <= 0.0:
        return 0.5
    return sum(i * v for i, v in enumerate(bands)) / total / (len(bands) - 1)
