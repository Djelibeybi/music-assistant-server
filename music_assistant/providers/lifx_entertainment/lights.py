"""Discovery and connections to the LIFX lights, through lifx-async."""

from __future__ import annotations

import asyncio
import colorsys
import logging
from contextlib import aclosing, suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from lifx import Connectivity, Device, LifxError, MultiZoneLight
from lifx.const import LIFX_UDP_PORT
from lifx.network.address import validate_address
from lifx.network.discovery import discover_devices

from .models import LifxDevice, ZoneSupport

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from lifx import Light
    from lifx.network.discovery import DiscoveredDevice

LOGGER = logging.getLogger(__name__)

# Overall limit for one request, retransmits included.
REQUEST_TIMEOUT_S = 2.0
DEFAULT_KELVIN = 3500


class Unsupported(StrEnum):
    """Why a light can't be synced to music."""

    # Without extended multizone a strip can't take a frame per update.
    LEGACY_MULTIZONE = "legacy_multizone"
    # A Thread mesh is not built for a steady stream of frames.
    THREAD = "thread"


@dataclass
class FoundLight:
    """A light found by :func:`discover_lights`."""

    device: LifxDevice
    unsupported: Unsupported | None = None


def is_valid_address(address: str) -> bool:
    """
    Return True when the address is an IPv4 or IPv6 literal lifx-async can send to.

    :param address: The address to check.
    """
    try:
        validate_address(address)
    except ValueError:
        return False
    return True


async def locate_lights(
    *,
    timeout: float,
    broadcast: str,
    hosts: list[str],
    port: int = LIFX_UDP_PORT,
) -> list[DiscoveredDevice]:
    """
    Find the lights that answer on the network, without querying them.

    :param timeout: Discovery window in seconds.
    :param broadcast: Address the discovery broadcast is sent to.
    :param hosts: Light addresses to probe directly, for networks that drop broadcasts.
    :param port: UDP port the lights listen on.
    """
    found: dict[str, DiscoveredDevice] = {}

    async def _sweep(address: str) -> None:
        sweep = discover_devices(
            timeout=timeout,
            broadcast_address=address,
            port=port,
            device_timeout=REQUEST_TIMEOUT_S,
        )
        async with aclosing(sweep):
            async for discovered in sweep:
                # The broadcast and a direct probe can both find a light.
                found.setdefault(discovered.serial, discovered)

    await _sweep_all(_sweep, broadcast, hosts)
    return list(found.values())


async def discover_lights(
    *,
    timeout: float,
    broadcast: str,
    hosts: list[str],
    port: int = LIFX_UDP_PORT,
) -> list[FoundLight]:
    """
    Find the lights on the network and read their details, sorted by label.

    :param timeout: Discovery window in seconds.
    :param broadcast: Address the discovery broadcast is sent to.
    :param hosts: Light addresses to probe directly, for networks that drop broadcasts.
    :param port: UDP port the lights listen on.
    """
    located = await locate_lights(timeout=timeout, broadcast=broadcast, hosts=hosts, port=port)
    results = await asyncio.gather(*(_describe(d) for d in located))
    lights = [light for light in results if light is not None]
    return sorted(lights, key=lambda f: (f.device.label.lower(), f.device.serial))


def rgb16_to_hsbk(
    red: int, green: int, blue: int, kelvin: int = DEFAULT_KELVIN
) -> tuple[int, int, int, int]:
    """
    Convert 16-bit RGB to a protocol-ready HSBK tuple.

    Brightness carries the RGB value (max channel), so a dimmed RGB colour becomes a
    dimmed LIFX colour with the same hue and saturation.

    :param red: Red, 0-65535.
    :param green: Green, 0-65535.
    :param blue: Blue, 0-65535.
    :param kelvin: White point for low-saturation colours.
    """
    r = max(0, min(65535, red)) / 65535.0
    g = max(0, min(65535, green)) / 65535.0
    b = max(0, min(65535, blue)) / 65535.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (round(h * 65535) & 0xFFFF, round(s * 65535), round(v * 65535), kelvin)


class LightPool:
    """The lifx-async connection to every stored light, opened on first use."""

    def __init__(self) -> None:
        """Initialize the pool."""
        self._lights: dict[str, Light] = {}
        self._lock = asyncio.Lock()

    async def get(self, device: LifxDevice) -> Light:
        """
        Return the connection to a light, connecting first if needed.

        :param device: The stored light.
        """
        async with self._lock:
            light = self._lights.get(device.serial)
            if light is not None:
                light.update_address(device.host, device.port)
                return light
            light = await Device.connect(
                device.host,
                serial=device.serial,
                port=device.port,
                timeout=REQUEST_TIMEOUT_S,
            )
            self._lights[device.serial] = light
            return light

    def follow(self, serial: str, host: str, port: int) -> None:
        """
        Move an open connection to a light's new address.

        :param serial: The light's serial.
        :param host: The light's new address.
        :param port: The light's new UDP port.
        """
        light = self._lights.get(serial)
        if light is not None:
            light.update_address(host, port)

    async def close(self) -> None:
        """Close every connection."""
        async with self._lock:
            lights, self._lights = list(self._lights.values()), {}
        for light in lights:
            light.animator.close()
            with suppress(Exception):
                await light.close()


async def _sweep_all(
    sweep: Callable[[str], Awaitable[None]], broadcast: str, hosts: list[str]
) -> None:
    """Run the broadcast and every direct probe at once; only a failed broadcast raises."""
    unique_hosts = [h for h in dict.fromkeys(hosts) if h != broadcast]
    results = await asyncio.gather(
        sweep(broadcast), *(sweep(host) for host in unique_hosts), return_exceptions=True
    )
    for host, result in zip(unique_hosts, results[1:], strict=True):
        if isinstance(result, Exception):
            LOGGER.debug("Probing LIFX light at %s failed: %s", host, result)
    if isinstance(results[0], BaseException):
        raise results[0]


async def _describe(discovered: DiscoveredDevice) -> FoundLight | None:
    """Read a light's label, group, product and zones, and whether it can be synced."""
    try:
        light = await discovered.create_device()  # type: ignore[no-untyped-call]
    except LifxError as err:
        LOGGER.debug("LIFX device %s is not a supported light: %s", discovered.serial, err)
        return None
    if light is None:
        return None
    try:
        _color, _power, label = await light.get_color()
        device = LifxDevice(serial=light.serial, host=light.ip, port=light.port, label=label or "")
        with suppress(LifxError):
            group = await light.get_group()
            device.group_id, device.group_label = group.uuid, group.label
        if light.version is not None:
            device.product_id = light.version.product
        if isinstance(light, MultiZoneLight):
            device.zone_count = await light.get_zone_count()
            extended = light.capabilities is not None and light.capabilities.has_extended_multizone
            device.zone_support = ZoneSupport.EXTENDED if extended else ZoneSupport.LEGACY
        # Every reply so far carried the light's own report of its radio.
        if light.connectivity is Connectivity.THREAD:
            return FoundLight(device, Unsupported.THREAD)
        if device.zone_support == ZoneSupport.LEGACY:
            return FoundLight(device, Unsupported.LEGACY_MULTIZONE)
        return FoundLight(device)
    except LifxError as err:
        LOGGER.debug("LIFX light %s at %s did not answer: %s", light.serial, light.ip, err)
        return None
    finally:
        with suppress(Exception):
            await light.close()
