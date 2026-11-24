"""A minimal in-process fake LIFX light listening on localhost UDP."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lifx.protocol import packets
from lifx.protocol.header import LifxHeader
from lifx.protocol.packets import get_packet_class
from lifx.protocol.protocol_types import DeviceService, LightHsbk

if TYPE_CHECKING:
    from lifx.protocol.base import Packet

# A bulb, and a strip whose extended multizone support depends on its firmware (2.77+).
BULB_PRODUCT = 27
STRIP_PRODUCT = 32
EXTENDED_FIRMWARE = (3, 70)
LEGACY_FIRMWARE = (2, 70)


@dataclass(frozen=True)
class Hsbk:
    """A protocol-level colour, comparable by value."""

    hue: int
    saturation: int
    brightness: int
    kelvin: int

    @classmethod
    def of(cls, color: LightHsbk) -> Hsbk:
        """Convert from a packet field."""
        return cls(color.hue, color.saturation, color.brightness, color.kelvin)

    def packed(self) -> LightHsbk:
        """Convert to a packet field."""
        return LightHsbk(self.hue, self.saturation, self.brightness, self.kelvin)


class FakeLifxDevice(asyncio.DatagramProtocol):
    """Answers the subset of the LAN protocol lifx-async uses, and records every packet."""

    def __init__(
        self,
        serial: str,
        label: str,
        *,
        group: str = "Living room",
        group_id: bytes = b"\x11" * 16,
        zones: int = 0,
        power: int = 65535,
        color: Hsbk | None = None,
        firmware: tuple[int, int] = EXTENDED_FIRMWARE,
        thread: bool = False,
    ) -> None:
        """Describe the light; call `start` to listen."""
        self.serial = serial
        self.label = label
        self.group = group
        self.group_id = group_id
        self.zone_count = zones
        self.power = power
        self.color = color or Hsbk(1000, 2000, 3000, 3500)
        self.zones = [self.color] * zones
        self.firmware = firmware
        self.thread = thread
        self.received: list[tuple[LifxHeader, object]] = []
        self.port = 0
        self._transport: asyncio.DatagramTransport | None = None

    async def start(self) -> None:
        """Listen on a free localhost UDP port."""
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: self, local_addr=("127.0.0.1", 0)
        )
        self._transport = transport
        self.port = transport.get_extra_info("sockname")[1]

    def close(self) -> None:
        """Stop listening."""
        if self._transport is not None:
            self._transport.close()

    def of_type(self, packet_class: type) -> list[tuple[LifxHeader, object]]:
        """
        Return the received packets of one type.

        :param packet_class: The packet class, e.g. ``packets.Light.SetColor``.
        """
        return [(h, p) for h, p in self.received if isinstance(p, packet_class)]

    def datagram_received(self, data: bytes, addr: tuple[str | int, int]) -> None:
        """Record a packet addressed to this light and answer it."""
        header = LifxHeader.unpack(data[: LifxHeader.HEADER_SIZE])
        if not header.tagged and header.target_serial.hex() != self.serial:
            return
        packet_class = get_packet_class(header.pkt_type)
        if packet_class is None:
            return
        packet = packet_class.unpack(data[LifxHeader.HEADER_SIZE : header.size])
        self.received.append((header, packet))
        reply = self._handle(packet)
        if header.ack_required:
            self._reply(header, addr, packets.Device.Acknowledgement())
        if reply is not None and (header.res_required or packet_class.PKT_TYPE == 2):
            self._reply(header, addr, reply)

    def _reply(self, header: LifxHeader, addr: tuple[str | int, int], packet: Packet) -> None:
        assert self._transport is not None
        payload = packet.pack()
        reply = LifxHeader.create(
            pkt_type=packet.PKT_TYPE,
            source=header.source,
            target=bytes.fromhex(self.serial),
            res_required=False,
            sequence=header.sequence,
            payload_size=len(payload),
        )
        reply.thread_connection = self.thread
        self._transport.sendto(reply.pack() + payload, (str(addr[0]), int(addr[1])))

    def _handle(self, packet: Packet) -> Packet | None:
        match packet:
            case packets.Device.GetService():
                return packets.Device.StateService(service=DeviceService.UDP, port=self.port)
            case packets.Device.GetVersion():
                product = STRIP_PRODUCT if self.zone_count else BULB_PRODUCT
                return packets.Device.StateVersion(vendor=1, product=product)
            case packets.Device.GetHostFirmware():
                major, minor = self.firmware
                return packets.Device.StateHostFirmware(
                    build=0, version_minor=minor, version_major=major
                )
            case packets.Device.GetLabel():
                return packets.Device.StateLabel(label=self.label)
            case packets.Device.GetGroup():
                return packets.Device.StateGroup(
                    group=self.group_id, label=self.group, updated_at=0
                )
            case packets.Device.GetPower() | packets.Light.GetPower():
                return packets.Light.StatePower(level=self.power)
            case packets.Device.SetPower(level=level) | packets.Light.SetPower(level=level):
                self.power = level
                return packets.Light.StatePower(level=self.power)
            case packets.Light.GetColor():
                return packets.Light.StateColor(
                    color=self.color.packed(), power=self.power, label=self.label
                )
            case packets.Light.SetColor(color=color):
                self.color = Hsbk.of(color)
                self.zones = [self.color] * self.zone_count
                return None
        return self._handle_zones(packet)

    def _handle_zones(self, packet: Packet) -> Packet | None:
        if not self.zone_count:
            return packets.Device.StateUnhandled(unhandled_type=packet.PKT_TYPE)
        match packet:
            case packets.MultiZone.GetColorZones(start_index=start):
                window = (self.zones[start:] + [self.color] * 8)[:8]
                return packets.MultiZone.StateMultiZone(
                    count=self.zone_count, index=start, colors=[c.packed() for c in window]
                )
            case packets.MultiZone.GetExtendedColorZones():
                colors = [c.packed() for c in self.zones]
                colors += [LightHsbk(0, 0, 0, 0)] * (82 - len(colors))
                return packets.MultiZone.StateExtendedColorZones(
                    count=self.zone_count, index=0, colors_count=self.zone_count, colors=colors
                )
            case packets.MultiZone.SetExtendedColorZones(
                index=index, colors_count=count, colors=colors
            ):
                for offset in range(count):
                    if index + offset < self.zone_count:
                        self.zones[index + offset] = Hsbk.of(colors[offset])
                return None
        return None
