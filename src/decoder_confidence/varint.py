"""VarInt encoding for obs_flip_idx binary files.

Per-shot binary blob format:
  varint(n_indices)
  varint(idx[0])           # absolute index
  varint(idx[1]-idx[0])   # delta from previous
  ...

Final file format (obs_flip_idx_batch={n}.bin):
  [num_shots : uint64 little-endian]
  [blob for shot 0]
  [blob for shot 1]
  ...

VarInt uses unsigned LEB128: 7 data bits per byte, MSB is continuation flag.
"""
from __future__ import annotations

import struct
from pathlib import Path


def encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError(f"VarInt value must be non-negative, got {value}")
    buf = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            buf.append(byte | 0x80)
        else:
            buf.append(byte)
            break
    return bytes(buf)


def decode_varint(data: bytes | bytearray | memoryview, offset: int) -> tuple[int, int]:
    """Return (value, new_offset)."""
    result = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError(f"Unexpected end of data at offset {offset}")
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            break
    return result, offset


def encode_obs_flip_shot(indices: list[int]) -> bytes:
    """Encode one shot's observable flip indices as a VarInt blob."""
    buf = bytearray(encode_varint(len(indices)))
    prev = 0
    for idx in sorted(indices):
        buf.extend(encode_varint(idx - prev))
        prev = idx
    return bytes(buf)


def decode_obs_flip_shot(
    data: bytes | bytearray | memoryview, offset: int
) -> tuple[list[int], int]:
    """Decode one shot's blob. Returns (indices, new_offset)."""
    n, offset = decode_varint(data, offset)
    indices: list[int] = []
    prev = 0
    for _ in range(n):
        delta, offset = decode_varint(data, offset)
        prev += delta
        indices.append(prev)
    return indices, offset


def write_obs_flip_idx_file(path: Path | str, shot_blobs: list[bytes]) -> None:
    """Write the final obs_flip_idx binary file from pre-encoded per-shot blobs."""
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(shot_blobs)))
        for blob in shot_blobs:
            fh.write(blob)


def read_obs_flip_idx_file(path: Path | str) -> list[list[int]]:
    """Read an obs_flip_idx binary file. Returns a list of per-shot index lists."""
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 8:
        raise ValueError(f"File too short ({len(data)} bytes), expected at least 8")
    (num_shots,) = struct.unpack_from("<Q", data, 0)
    offset = 8
    result: list[list[int]] = []
    for _ in range(num_shots):
        indices, offset = decode_obs_flip_shot(data, offset)
        result.append(indices)
    if offset != len(data):
        raise ValueError(
            f"Trailing bytes in obs_flip_idx file: {len(data) - offset} unexpected bytes"
        )
    return result
