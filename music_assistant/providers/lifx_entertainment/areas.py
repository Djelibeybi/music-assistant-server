"""
Build entertainment areas (and their light channels) from the stored lights.

The area ids feed the Sendspin client ids, which become the player ids, so they must
stay stable across releases.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from .models import EntertainmentArea, LifxDevice, LightChannel

# A multizone strip is split into at most this many channels, so the analyzer's
# per-channel spectrum bands stay musically meaningful on long strips.
DEFAULT_MAX_SEGMENTS = 8
# A matrix is split into this many (columns, rows) of channels for the analyzer, whatever
# its size: enough to make it multicoloured without crowding the analyzer's filters.
MATRIX_GRID = (3, 3)


class Grouping(StrEnum):
    """How lights are combined into entertainment areas."""

    # One area per LIFX group (the rooms configured in the LIFX app).
    GROUP = "group"
    # One area containing every light.
    ALL = "all"
    # One area per light.
    LIGHT = "light"


def build_areas(
    devices: list[LifxDevice],
    grouping: Grouping | str = Grouping.GROUP,
    *,
    all_name: str = "All lights",
    max_segments: int = DEFAULT_MAX_SEGMENTS,
) -> list[EntertainmentArea]:
    """
    Combine lights into entertainment areas, sorted by name.

    :param devices: The lights to combine.
    :param grouping: How the lights are combined.
    :param all_name: Name of the area holding every light, and of lights without a group.
    :param max_segments: Maximum number of channels a multizone strip is split into.
    """
    grouping = Grouping(grouping)
    ordered = sorted(devices, key=lambda d: (d.label.lower(), d.serial))
    buckets: dict[str, tuple[str, list[LifxDevice]]] = {}
    for device in ordered:
        if grouping == Grouping.ALL:
            key, name = "all", all_name
        elif grouping == Grouping.LIGHT:
            key, name = f"light-{device.serial}", device.label or device.serial
        else:
            group = device.group_id or "nogroup"
            key, name = f"group-{group}", device.group_label or all_name
        buckets.setdefault(key, (name, []))[1].append(device)
    areas = [
        EntertainmentArea(
            id=key,
            name=name,
            devices=members,
            channels=build_channels(members, max_segments=max_segments),
        )
        for key, (name, members) in buckets.items()
    ]
    return sorted(areas, key=lambda a: a.name.lower())


def build_channels(
    devices: list[LifxDevice], *, max_segments: int = DEFAULT_MAX_SEGMENTS
) -> list[LightChannel]:
    """
    Create the light channels for a set of lights, laid out left to right.

    :param devices: The lights of one area, in display order.
    :param max_segments: Maximum number of channels a multizone strip is split into.
    """
    # One slot per position along the x axis; a matrix fills a slot per grid column,
    # with its rows stacked on the z axis so the analyzer gives them their own bands.
    slots: list[list[_Spec]] = []
    for device in devices:
        name = device.label or device.serial
        if device.is_matrix:
            columns, rows = MATRIX_GRID
            for column in range(columns):
                slots.append(
                    [
                        _Spec(
                            device,
                            name=f"{name} #{row * columns + column + 1}",
                            column=column,
                            row=row,
                        )
                        for row in range(rows)
                    ]
                )
        elif device.is_multizone:
            segments = max(1, min(max_segments, device.zone_count))
            for segment in range(segments):
                start = segment * device.zone_count // segments
                end = (segment + 1) * device.zone_count // segments
                slots.append([_Spec(device, name=f"{name} #{segment + 1}", start=start, end=end)])
        else:
            slots.append([_Spec(device, name=name)])
    count = len(slots)
    channels: list[LightChannel] = []
    # Positions along the x axis give the analyzer a spatial axis for its colour gradient.
    for index, slot in enumerate(slots):
        x = 0.0 if count == 1 else -1.0 + 2.0 * index / (count - 1)
        for spec in slot:
            channels.append(
                LightChannel(
                    channel_id=len(channels),
                    service_id=spec.device.serial,
                    name=spec.name,
                    position=(x, 0.0, spec.z),
                    zone_start=spec.start,
                    zone_end=spec.end,
                    column=spec.column,
                    row=spec.row,
                )
            )
    return channels


def area_client_id(area_id: str, namespace: str = "") -> str:
    """
    Return a short stable identifier for an area.

    :param area_id: The area id from :func:`build_areas`.
    :param namespace: Keeps the ids of several provider instances apart.
    """
    return hashlib.sha1(f"{namespace}:{area_id}".encode(), usedforsecurity=False).hexdigest()[:16]


@dataclass(frozen=True)
class _Spec:
    """One channel to create, before its position along the x axis is known."""

    device: LifxDevice
    name: str
    start: int = 0
    end: int = 1
    column: int = 0
    row: int = 0

    @property
    def z(self) -> float:
        """Height of a matrix row in [-1, 1], top row highest; 0 for other lights."""
        rows = MATRIX_GRID[1]
        if not self.device.is_matrix or rows < 2:
            return 0.0
        return 1.0 - 2.0 * self.row / (rows - 1)
