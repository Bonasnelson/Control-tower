"""Start the demo dashboard: FastAPI backend (uvicorn) + Vite dev server, one command.

    python scripts/run_dashboard.py            # http://127.0.0.1:5173 ; Ctrl+C stops both
    python scripts/run_dashboard.py --open     # also open the browser
    python scripts/run_dashboard.py --check    # start, verify /api/health through the Vite proxy, stop

Needs the backend packages (``python -m pip install -r app/requirements.txt``) and the
front-end packages (``npm ci`` in app/web; ``--install`` runs it if node_modules is missing).
The dashboard is read-only: it serves files from configs/ and results/; nothing is trained.
Works on Windows, macOS and Linux (each server runs in its own process group so that
Ctrl+C stops the whole tree, including the node process npm starts).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "app" / "web"
API_DIR = ROOT / "app" / "api"
WINDOWS = os.name == "nt"


def start(cmd: list[str], cwd: Path, env: dict, log: Path | None) -> subprocess.Popen:
    out = open(log, "w", encoding="utf-8") if log else None  # a file, never an unread PIPE (no hang)
    kw: dict = {"cwd": cwd, "env": env, "stdout": out or None, "stderr": subprocess.STDOUT if out else None}
    if WINDOWS:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(cmd, **kw)


def stop(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    if WINDOWS:  # kill the whole tree (npm -> node)
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()


def wait_json(url: str, timeout: float, procs: list[subprocess.Popen]) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if any(p.poll() is not None for p in procs):
            raise RuntimeError("a server exited during start-up (see the log above or --log-dir)")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"no answer from {url} within {timeout:.0f}s")


def _interrupt(*_):
    raise KeyboardInterrupt


def main() -> int:
    # Ctrl+Break (Windows) and SIGTERM (POSIX) stop both servers just like Ctrl+C.
    signal.signal(signal.SIGBREAK if WINDOWS else signal.SIGTERM, _interrupt)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--web-port", type=int, default=5173)
    ap.add_argument("--open", action="store_true", help="open the dashboard in the default browser")
    ap.add_argument("--install", action="store_true", help="run `npm ci` first if app/web/node_modules is missing")
    ap.add_argument("--check", action="store_true", help="start both, check /api/health through the proxy, then stop")
    ap.add_argument("--log-dir", type=Path, default=None, help="write server output to files here instead of the console")
    ap.add_argument("--timeout", type=float, default=60)
    a = ap.parse_args()

    for mod in ("fastapi", "uvicorn"):
        if importlib.util.find_spec(mod) is None:
            print(f"Missing Python package '{mod}'. Install with: {sys.executable} -m pip install -r app/requirements.txt")
            return 1
    npm = shutil.which("npm")
    if npm is None:
        print("npm not found. Install Node.js 18+ (https://nodejs.org) and run this script again.")
        return 1
    if not (WEB / "node_modules").exists():
        if not a.install:
            print("app/web/node_modules is missing. Run `npm ci` in app/web, or pass --install.")
            return 1
        subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=WEB, check=True)

    env = {**os.environ, "CT_API_URL": f"http://127.0.0.1:{a.api_port}", "PYTHONUNBUFFERED": "1"}
    logs = (a.log_dir / "api.log", a.log_dir / "web.log") if a.log_dir else (None, None)
    if a.log_dir:
        a.log_dir.mkdir(parents=True, exist_ok=True)
    api = start([sys.executable, "-m", "uvicorn", "main:app", "--app-dir", str(API_DIR),
                 "--host", "127.0.0.1", "--port", str(a.api_port)], ROOT, env, logs[0])
    web = start([npm, "run", "dev", "--", "--port", str(a.web_port), "--strictPort", "--host", "127.0.0.1"],
                WEB, env, logs[1])
    procs = [api, web]
    url = f"http://127.0.0.1:{a.web_port}"
    try:
        wait_json(f"http://127.0.0.1:{a.api_port}/api/health", a.timeout, procs)
        health = wait_json(f"http://127.0.0.1:{a.web_port}/api/health", a.timeout, procs)  # through the Vite proxy
        print(f"\nControl Tower dashboard: {url}\n  API  http://127.0.0.1:{a.api_port} (health via proxy: {health.get('status')})"
              f"\n  Data {health.get('root')}\n  Press Ctrl+C to stop both servers.\n", flush=True)
        if a.check:
            return 0 if health.get("status") == "ok" else 1
        if a.open:
            webbrowser.open(url)
        while all(p.poll() is None for p in procs):
            time.sleep(0.5)
        print("A server stopped; shutting down the other one.")
        return 1
    except KeyboardInterrupt:
        print("\nStopping…")
        return 0
    except (RuntimeError, TimeoutError) as e:
        print(f"Start-up failed: {e}")
        return 1
    finally:
        for p in procs:
            stop(p)
        print("Both servers stopped.")


if __name__ == "__main__":
    sys.exit(main())
