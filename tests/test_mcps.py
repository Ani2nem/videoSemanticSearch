"""
tests/test_mcps.py

Integration tests for all five MCP servers.

Each test:
  1. Starts the MCP server in-process on its assigned port
  2. Sends a valid FeatureRequest with a minimal real payload
  3. Asserts FeatureResponse.success == True
  4. Asserts FeatureResponse.result_json is valid parseable JSON
  5. Asserts FeatureResponse.confidence is in [0.0, 1.0]

Run:
    pytest tests/test_mcps.py -v

Note: Tests require the protobuf stubs to be generated first:
    python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/mcp.proto
"""

from __future__ import annotations

import asyncio
import io
import json
import struct
import wave
import pytest
import pytest_asyncio

import grpc
import grpc.aio

import mcp_pb2
import mcp_pb2_grpc

from mcps.hair_color.server   import HairColorServicer
from mcps.body_build.server   import BodyBuildServicer
from mcps.people_count.server import PeopleCountServicer
from mcps.captions.server     import CaptionsServicer
from mcps.audio.server        import AudioServicer


# ---------------------------------------------------------------------------
# Minimal payload factories
# ---------------------------------------------------------------------------

def _minimal_png_bytes() -> bytes:
    """
    Returns a 1×1 pixel white PNG as bytes.
    Generated inline so the test has zero file-system dependencies.
    """
    import zlib

    def _pack_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc    = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1×1, 8-bit RGB
    ihdr = _pack_chunk(b"IHDR", ihdr_data)
    raw_row = b"\x00\xff\xff\xff"                              # filter byte + R G B
    idat = _pack_chunk(b"IDAT", zlib.compress(raw_row))
    iend = _pack_chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def _minimal_wav_bytes() -> bytes:
    """Returns ~0.1 s of silence as a valid WAV bytes object."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(16000)
        # 1600 frames = 0.1 s of silence
        wf.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Server fixture factory
# ---------------------------------------------------------------------------

async def _start_server(servicer, port: int):
    """Start an in-process gRPC server and yield it; stop after the test."""
    server = grpc.aio.server()
    mcp_pb2_grpc.add_MCPServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    yield server
    await server.stop(grace=0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def hair_color_server():
    async for s in _start_server(HairColorServicer(), 50151):  # offset ports to avoid conflicts
        yield s

@pytest_asyncio.fixture
async def body_build_server():
    async for s in _start_server(BodyBuildServicer(), 50152):
        yield s

@pytest_asyncio.fixture
async def people_count_server():
    async for s in _start_server(PeopleCountServicer(), 50153):
        yield s

@pytest_asyncio.fixture
async def captions_server():
    async for s in _start_server(CaptionsServicer(), 50154):
        yield s

@pytest_asyncio.fixture
async def audio_server():
    async for s in _start_server(AudioServicer(), 50155):
        yield s


# ---------------------------------------------------------------------------
# Shared assertion helper
# ---------------------------------------------------------------------------

async def _assert_valid_response(address: str, payload: bytes, payload_type: str) -> None:
    async with grpc.aio.insecure_channel(address) as channel:
        stub = mcp_pb2_grpc.MCPServiceStub(channel)
        request = mcp_pb2.FeatureRequest(
            payload=payload,
            payload_type=payload_type,
            source_id="test-source-001",
        )
        response = await stub.ExtractFeatures(request)

    assert response.success is True, (
        f"Expected success=True but got success=False. Error: {response.error_message}"
    )

    assert response.result_json, "result_json must not be empty"
    parsed = json.loads(response.result_json)
    assert isinstance(parsed, dict), "result_json must deserialise to a dict"

    assert 0.0 <= response.confidence <= 1.0, (
        f"confidence {response.confidence} is outside [0.0, 1.0]"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hair_color_mcp(hair_color_server):
    """hair_color MCP returns a valid FeatureResponse for a tiny PNG."""
    await _assert_valid_response(
        "localhost:50151",
        _minimal_png_bytes(),
        "image",
    )


@pytest.mark.asyncio
async def test_body_build_mcp(body_build_server):
    """body_build MCP returns a valid FeatureResponse for a tiny PNG."""
    await _assert_valid_response(
        "localhost:50152",
        _minimal_png_bytes(),
        "image",
    )


@pytest.mark.asyncio
async def test_people_count_mcp(people_count_server):
    """people_count MCP returns a valid FeatureResponse for a tiny PNG."""
    await _assert_valid_response(
        "localhost:50153",
        _minimal_png_bytes(),
        "image",
    )


@pytest.mark.asyncio
async def test_captions_mcp(captions_server):
    """captions MCP returns a valid FeatureResponse for a tiny PNG."""
    await _assert_valid_response(
        "localhost:50154",
        _minimal_png_bytes(),
        "image",
    )


@pytest.mark.asyncio
async def test_audio_mcp(audio_server):
    """audio MCP returns a valid FeatureResponse for a short WAV clip."""
    await _assert_valid_response(
        "localhost:50155",
        _minimal_wav_bytes(),
        "audio",
    )
