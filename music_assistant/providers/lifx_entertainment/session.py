"""Stream colour frames to the lights of one entertainment area."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from lifx import HSBK, LifxError

from .lights import DEFAULT_KELVIN, rgb16_to_hsbk
from .models import ZoneSupport

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lifx import Light
    from lifx.animation import Animator
    from lifx.effects import DeviceStateManager, PreState

    from music_assistant.providers.sendspin_lights.models import LightColor

    from .lights import LightPool
    from .models import EntertainmentArea, LifxDevice

LOGGER = logging.getLogger(__name__)

DEFAULT_TRANSITION_MS = 50
# Unchanged frames are suppressed, but re-sent this often so a dropped UDP packet
# never leaves a light stuck on a stale colour.
_KEEPALIVE_S = 1.0

_Frame = list[tuple[int, int, int, int]]


class EntertainmentSession:
    """
    Stream colours to every light of an entertainment area.

    :meth:`start` captures each light's state, :meth:`aclose` restores it.
    """

    def __init__(
        self,
        pool: LightPool,
        state_manager: DeviceStateManager,
        area: EntertainmentArea,
        *,
        transition_ms: int = DEFAULT_TRANSITION_MS,
        power_on: bool = True,
        kelvin: int = DEFAULT_KELVIN,
    ) -> None:
        """
        Initialize the session (no network I/O until :meth:`start`).

        :param pool: The connections to the lights.
        :param state_manager: Captures and restores the lights' state.
        :param area: The lights to stream to.
        :param transition_ms: Fade time of each frame; smooths over frame jitter.
        :param power_on: Switch on lights that are off when the session starts.
        :param kelvin: White point for low-saturation colours.
        """
        self._pool = pool
        self._state_manager = state_manager
        self.area = area
        self._transition_ms = transition_ms
        self._power_on = power_on
        self._kelvin = kelvin
        self._channels = {ch.channel_id: ch for ch in area.channels}
        self._devices = {d.serial: d for d in area.devices}
        self._lights: dict[str, Light] = {}
        self._snapshots: dict[str, PreState] = {}
        self._animators: dict[str, Animator] = {}
        self._zone_frames: dict[str, _Frame] = {}
        self._last_frame: dict[str, _Frame] = {}
        self._last_sent: dict[str, float] = {}
        self._streaming = False
        self._lock = asyncio.Lock()

    @property
    def is_streaming(self) -> bool:
        """Return True while the session is active."""
        return self._streaming

    @property
    def transition_ms(self) -> int:
        """Return the fade time of each frame."""
        return self._transition_ms

    @transition_ms.setter
    def transition_ms(self, value: int) -> None:
        """Change the fade time of each frame, also while streaming."""
        self._transition_ms = value
        for animator in self._animators.values():
            animator.duration_ms = value

    async def start(self) -> None:
        """
        Capture every light's state, switch lights on and begin accepting frames.

        :raises ConnectionError: When no light of the area can be reached.
        """
        async with self._lock:
            if self._streaming:
                return
            devices = list(self._devices.values())
            results = await asyncio.gather(
                *(self._prepare_device(d) for d in devices), return_exceptions=True
            )
            ready = 0
            for device, result in zip(devices, results, strict=True):
                if isinstance(result, BaseException):
                    LOGGER.warning(
                        "LIFX light %s (%s) not ready: %s", device.label, device.host, result
                    )
                    continue
                ready += 1
            if devices and ready == 0:
                msg = f"No LIFX light of area '{self.area.name}' is reachable"
                raise ConnectionError(msg)
            self._last_frame.clear()
            self._last_sent.clear()
            self._zone_frames = {
                serial: [(0, 0, 0, self._kelvin)] * animator.pixel_count
                for serial, animator in self._animators.items()
                if self._devices[serial].is_multizone
            }
            self._streaming = True
            LOGGER.info(
                "LIFX session started for area '%s' (%d/%d lights)",
                self.area.name,
                ready,
                len(devices),
            )

    def send(self, colors: Iterable[LightColor]) -> None:
        """
        Push a frame of channel colours to the lights.

        Non-blocking and safe to call from the event loop; a no-op when not streaming.

        :param colors: The colour of each channel that changed.
        """
        if not self._streaming:
            return
        per_device: dict[str, list[tuple[int, int, tuple[int, int, int, int]]]] = {}
        for color in colors:
            channel = self._channels.get(color.channel_id)
            if channel is None or channel.service_id not in self._animators:
                continue
            hsbk = rgb16_to_hsbk(color.red, color.green, color.blue, self._kelvin)
            per_device.setdefault(channel.service_id, []).append(
                (channel.zone_start, channel.zone_end, hsbk)
            )
        now = time.monotonic()
        for serial, updates in per_device.items():
            animator = self._animators[serial]
            frame = self._build_frame(self._devices[serial], animator, updates)
            self._send_frame(serial, animator, frame, now)

    async def aclose(self) -> None:
        """Stop streaming and restore the captured light state."""
        async with self._lock:
            # A start that failed or was cancelled halfway may already have switched
            # lights on, so restore whatever was captured even if never streaming.
            if not self._streaming and not self._snapshots:
                return
            self._streaming = False
            self._animators.clear()
            await asyncio.gather(
                *(
                    self._restore_device(self._lights[serial], snapshot)
                    for serial, snapshot in self._snapshots.items()
                ),
                return_exceptions=True,
            )
            self._snapshots.clear()
            LOGGER.info("LIFX session stopped for area '%s'", self.area.name)

    async def _prepare_device(self, device: LifxDevice) -> None:
        if device.zone_support == ZoneSupport.LEGACY:
            msg = "the strip has no extended multizone support"
            raise LifxError(msg)
        light = await self._pool.get(device)
        # A reply carries the light's own report of its radio, so read before preparing:
        # the Animator then refuses a Thread light (or a strip without extended
        # multizone) before the light is touched.
        await light.get_power()
        animator = await light.animator.prepare()
        animator.duration_ms = self._transition_ms
        # Keep the first capture across start retries: by then this session may already
        # have switched the light on, which must not become the state to restore.
        snapshot = self._snapshots.get(device.serial)
        if snapshot is None:
            snapshot = await self._state_manager.capture_state(light)
            self._lights[device.serial] = light
            self._snapshots[device.serial] = snapshot
        if not snapshot.power and self._power_on:
            # Go dark before powering up so the light doesn't flash its old colour.
            await light.set_color(HSBK(0, 0, 0, self._kelvin))
            await light.set_power(True)
        self._animators[device.serial] = animator

    async def _restore_device(self, light: Light, snapshot: PreState) -> None:
        try:
            await self._state_manager.restore_state(light, snapshot)
        except (LifxError, OSError) as err:
            LOGGER.warning("Failed to restore LIFX light %s: %s", light.label, err)

    def _build_frame(
        self,
        device: LifxDevice,
        animator: Animator,
        updates: list[tuple[int, int, tuple[int, int, int, int]]],
    ) -> _Frame:
        if not device.is_multizone:
            # A bulb, tile or ceiling light has a single channel and shows one colour;
            # if several channels map to it, average them.
            color = updates[0][2] if len(updates) == 1 else _average([u[2] for u in updates])
            return [color] * animator.pixel_count
        zones = self._zone_frames.setdefault(
            device.serial, [(0, 0, 0, self._kelvin)] * animator.pixel_count
        )
        for start, end, color in updates:
            for index in range(max(0, start), min(len(zones), end)):
                zones[index] = color
        return list(zones)

    def _send_frame(self, serial: str, animator: Animator, frame: _Frame, now: float) -> None:
        unchanged = self._last_frame.get(serial) == frame
        if unchanged and now - self._last_sent.get(serial, 0.0) < _KEEPALIVE_S:
            return
        try:
            stats = animator.send_frame(frame)
        except (LifxError, OSError) as err:
            LOGGER.debug("Sending a frame to LIFX light %s failed: %s", serial, err)
            return
        # A light that is behind on acknowledgements drops the frame; retry it next time.
        if stats.gated:
            return
        self._last_frame[serial] = frame
        self._last_sent[serial] = now


def _average(colors: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """Average colours component-wise (only used when several channels share one light)."""
    count = len(colors)
    return (
        sum(c[0] for c in colors) // count,
        sum(c[1] for c in colors) // count,
        sum(c[2] for c in colors) // count,
        colors[0][3],
    )
