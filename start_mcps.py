"""
start_mcps.py — unified launcher for all Video Insight Engine MCP servers.

Usage
-----
    python start_mcps.py                              # start all 5 servers
    python start_mcps.py --stop                       # stop all running servers
    python start_mcps.py --servers hair_color race age # start specific servers
    python start_mcps.py --no-wait                    # launch without readiness probe
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve the Python interpreter to use for subprocesses.
# Use the venv Python so servers get the project's packages regardless of
# how this script was invoked.
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_VENV_DIR = _HERE / "venv"

# Re-exec this script inside the venv if we're not already running in it.
# Compares sys.prefix (the active env root) against the project venv dir.
if _VENV_DIR.exists() and Path(sys.prefix).resolve() != _VENV_DIR.resolve():
    import os
    _vp = str(_VENV_DIR / "bin" / "python3")
    os.execv(_vp, [_vp] + sys.argv)

PYTHON = str(_VENV_DIR / "bin" / "python3") if _VENV_DIR.exists() else sys.executable

# ---------------------------------------------------------------------------
# Server registry — must stay in sync with agents/extraction_agent.py:MCP_ADDRESSES
# ---------------------------------------------------------------------------

SERVERS: dict[str, dict] = {
    "hair_color":   {"module": "mcps.hair_color.server",   "port": 50051},
    "body_build":   {"module": "mcps.body_build.server",   "port": 50052},
    "people_count": {"module": "mcps.people_count.server", "port": 50053},
    "race":         {"module": "mcps.race.server",         "port": 50056},
    "age":          {"module": "mcps.age.server",          "port": 50057},
}

PID_FILE = Path(".mcps.pids")

# ANSI colours — one per server, cycling if more are added
_COLORS = ["\033[36m", "\033[33m", "\033[32m", "\033[35m", "\033[34m"]
_COLORS_MAP = {name: _COLORS[i % len(_COLORS)] for i, name in enumerate(SERVERS)}
RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"


def _tag(name: str) -> str:
    return f"{_COLORS_MAP[name]}[{name}]{RESET}"


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def _launch_servers(server_names: list[str]) -> dict[str, subprocess.Popen]:
    processes: dict[str, subprocess.Popen] = {}
    for name in server_names:
        info = SERVERS[name]
        print(f"{_tag(name)} Starting on port {info['port']}…")
        proc = subprocess.Popen(
            [PYTHON, "-m", info["module"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes[name] = proc

    PID_FILE.write_text(json.dumps({n: p.pid for n, p in processes.items()}, indent=2))
    return processes


# ---------------------------------------------------------------------------
# Log streaming
# ---------------------------------------------------------------------------

def _stream_logs(processes: dict[str, subprocess.Popen]) -> None:
    """Spawn one daemon thread per server to forward its output with a colour prefix."""
    def _reader(name: str, proc: subprocess.Popen) -> None:
        tag = _tag(name)
        for line in proc.stdout:  # type: ignore[union-attr]
            print(f"{tag} {line}", end="", flush=True)

    for name, proc in processes.items():
        t = threading.Thread(target=_reader, args=(name, proc), daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# Readiness probe
# ---------------------------------------------------------------------------

async def _probe_server(name: str, port: int, timeout: float = 30.0) -> bool:
    """Return True once the gRPC port is accepting connections, False on timeout."""
    import grpc.aio  # deferred so the script is importable without grpc installed

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            channel = grpc.aio.insecure_channel(f"localhost:{port}")
            await asyncio.wait_for(channel.channel_ready(), timeout=1.0)
            await channel.close()
            return True
        except Exception:
            await asyncio.sleep(0.5)
    return False


async def _wait_for_servers(server_names: list[str], timeout: float = 30.0) -> None:
    tasks = {
        name: asyncio.create_task(_probe_server(name, SERVERS[name]["port"], timeout))
        for name in server_names
    }

    pending = set(server_names)
    while pending:
        done = {n for n in list(pending) if tasks[n].done()}
        for name in done:
            pending.discard(name)
            ok = tasks[name].result()
            if ok:
                print(f"{_tag(name)} {GREEN}Ready{RESET} on port {SERVERS[name]['port']}")
            else:
                print(
                    f"{_tag(name)} {RED}WARNING:{RESET} did not become ready within "
                    f"{timeout:.0f}s — still loading models? The pipeline will retry."
                )
        if pending:
            await asyncio.sleep(0.25)

    print(f"\n{BOLD}All {len(server_names)} MCP server(s) launched.{RESET} "
          "Press Ctrl+C to stop.\n")


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------

def _stop_servers() -> None:
    if not PID_FILE.exists():
        print("No .mcps.pids file found — are servers running?")
        return

    pids: dict[str, int] = json.loads(PID_FILE.read_text())
    for name, pid in pids.items():
        tag = _tag(name) if name in _COLORS_MAP else f"[{name}]"
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"{tag} Stopped (PID {pid})")
        except ProcessLookupError:
            print(f"{tag} Already stopped (PID {pid})")

    PID_FILE.unlink(missing_ok=True)
    print("Done.")


# ---------------------------------------------------------------------------
# Shutdown hook (Ctrl+C / SIGTERM)
# ---------------------------------------------------------------------------

def _register_shutdown(processes: dict[str, subprocess.Popen]) -> None:
    def _handler(sig: int, frame: object) -> None:
        print(f"\n{BOLD}Shutting down all MCP servers…{RESET}")
        for proc in processes.values():
            proc.terminate()
        PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage Video Insight Engine MCP servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python start_mcps.py\n"
            "  python start_mcps.py --servers hair_color race\n"
            "  python start_mcps.py --stop\n"
        ),
    )
    parser.add_argument("--stop", action="store_true", help="Stop all running MCP servers")
    parser.add_argument(
        "--servers", nargs="+", choices=list(SERVERS),
        metavar="SERVER",
        help=f"Start only specific servers (choices: {', '.join(SERVERS)})",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Launch servers and return immediately without waiting for readiness",
    )
    args = parser.parse_args()

    if args.stop:
        _stop_servers()
        return

    server_names = args.servers or list(SERVERS)

    processes = _launch_servers(server_names)
    _stream_logs(processes)
    _register_shutdown(processes)

    if not args.no_wait:
        asyncio.run(_wait_for_servers(server_names))

    # Block until all processes exit (or Ctrl+C fires the shutdown hook)
    for proc in processes.values():
        proc.wait()


if __name__ == "__main__":
    main()
