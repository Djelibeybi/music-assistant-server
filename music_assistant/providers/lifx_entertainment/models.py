"""Data models for the LIFX Lights Sync provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from lifx.const import LIFX_UDP_PORT
from lifx.products import get_registry


class ZoneSupport(StrEnum):
    """How a light's colour zones are addressed."""

    NONE = "none"
    # Lights without extended multizone can't keep up with music sync, so they are refused.
    LEGACY = "legacy"
    EXTENDED = "extended"


@dataclass
class LifxDevice:
    """A LIFX light picked in the setup flow, as stored in the setup data."""

    serial: str
    host: str
    port: int = LIFX_UDP_PORT
    label: str = ""
    group_id: str = ""
    group_label: str = ""
    product_id: int = 0
    zone_support: ZoneSupport = ZoneSupport.NONE
    zone_count: int = 1

    @property
    def is_multizone(self) -> bool:
        """Return True when the light has individually addressable zones."""
        return self.zone_support != ZoneSupport.NONE and self.zone_count > 1

    @property
    def is_matrix(self) -> bool:
        """Return True when the light is a pixel matrix (Tile, Candle, Ceiling, ...)."""
        product = get_registry().get_product(self.product_id)
        return product is not None and product.has_matrix

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for the setup data."""
        return {
            "serial": self.serial,
            "host": self.host,
            "port": self.port,
            "label": self.label,
            "group_id": self.group_id,
            "group_label": self.group_label,
            "product_id": self.product_id,
            "zone_support": str(self.zone_support),
            "zone_count": self.zone_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LifxDevice:
        """
        Deserialize a dict produced by :meth:`to_dict`.

        :param data: The stored light.
        """
        return cls(
            serial=str(data["serial"]),
            host=str(data["host"]),
            port=int(data.get("port", LIFX_UDP_PORT)),
            label=str(data.get("label", "")),
            group_id=str(data.get("group_id", "")),
            group_label=str(data.get("group_label", "")),
            product_id=int(data.get("product_id", 0)),
            zone_support=ZoneSupport(data.get("zone_support", ZoneSupport.NONE)),
            zone_count=int(data.get("zone_count", 1)),
        )


@dataclass
class LightChannel:
    """
    A single light channel of an entertainment area.

    A plain bulb is one channel; a multizone strip is split into several channels, each
    driving a contiguous range of zones ``[zone_start, zone_end)``; a matrix is split into
    a grid of channels, each driving one cell ``(column, row)`` of the grid, row 0 at the top.
    """

    channel_id: int
    service_id: str
    name: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    zone_start: int = 0
    zone_end: int = 1
    column: int = 0
    row: int = 0


@dataclass
class EntertainmentArea:
    """A set of LIFX lights that react to music together."""

    id: str
    name: str
    devices: list[LifxDevice] = field(default_factory=list)
    channels: list[LightChannel] = field(default_factory=list)
