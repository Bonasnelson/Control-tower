"""Browser smoke tests for the demo dashboard (T11), Python Playwright + headless Chromium.

Two dashboards are started with scripts/run_dashboard.py (server output to log files):
* "real": this repository's configs/ and results/;
* "empty": CONTROLTOWER_ROOT = a temp root with configs/ but no results/ or data/, so every
  page must show its pending / blocked state instead of numbers.

Skips cleanly when Playwright (or its Chromium), fastapi/uvicorn, npm or app/web/node_modules
are missing. Screenshots go to $CT_SMOKE_SHOTS if set, else to a pytest temp dir.

    py -3.12 -m pytest -q -p no:cacheprovider tests/web/smoke_dashboard.py
    CT_SMOKE_SHOTS=<dir> py -3.12 -m pytest -q tests/web/smoke_dashboard.py

To add a page, append one entry to PAGES.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "app" / "web"

# id -> (expected h1, text that the "empty" root must show instead of data)
PAGES: dict[str, tuple[str, str]] = {
    "overview": ("Anomaly detection that keeps up with changing city payments", "does not exist yet"),
    "stream": ("The payment stream, E1 to E10", "does not exist yet"),
    "scoring": ("Which score finds the anomalies?", "does not exist yet"),
    "strategies": ("Which continual-learning strategy helps?", "does not exist yet"),
    "alerts": ("Records flagged in E10", "does not exist yet"),
    "elliptic": ("Elliptic Bitcoin transactions", "Blocked: the dataset is not in the repository"),
    # add new pages here (id -> expected h1, text shown when its results are missing)
}
WIDTHS = [(1440, 1000), (390, 844)]

for mod in ("playwright", "fastapi", "uvicorn"):
    if importlib.util.find_spec(mod) is None:
        pytest.skip(f"{mod} not installed (see app/requirements.txt / pip install playwright)", allow_module_level=True)
if shutil.which("npm") is None or not (WEB / "node_modules").exists():
    pytest.skip("npm or app/web/node_modules missing (run `npm ci` in app/web)", allow_module_level=True)

from playwright.sync_api import Browser, Page, sync_playwright  # noqa: E402


def _get(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


class Dashboard:
    """One run_dashboard.py process (API + Vite), stopped as a process group."""

    def __init__(self, name: str, api_port: int, web_port: int, log_dir: Path, root: Path | None = None):
        env = dict(os.environ)
        if root is not None:
            env["CONTROLTOWER_ROOT"] = str(root)
        kw: dict = {"cwd": ROOT, "env": env, "stdout": open(log_dir / f"{name}_launcher.log", "w"),
                    "stderr": subprocess.STDOUT}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        self.proc = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "run_dashboard.py"), "--api-port", str(api_port),
             "--web-port", str(web_port), "--log-dir", str(log_dir / name)], **kw)
        self.url = f"http://127.0.0.1:{web_port}"
        t0 = time.time()
        while time.time() - t0 < 90:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{name} dashboard exited early; see {log_dir}")
            if (_get(f"{self.url}/api/health") or {}).get("status") == "ok":
                return
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(f"{name} dashboard did not start; see {log_dir}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@pytest.fixture(scope="module")
def shots(tmp_path_factory) -> Path:
    d = Path(os.environ["CT_SMOKE_SHOTS"]) if os.environ.get("CT_SMOKE_SHOTS") else tmp_path_factory.mktemp("shots")
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def servers(tmp_path_factory):
    logs = tmp_path_factory.mktemp("dash_logs")
    empty = tmp_path_factory.mktemp("empty_root")
    shutil.copytree(ROOT / "configs", empty / "configs")  # cities/scenarios only; no results/, no data/
    (empty / "results").mkdir()
    real = Dashboard("real", 8011, 5191, logs)
    try:
        bare = Dashboard("empty", 8012, 5192, logs, root=empty)
    except Exception:
        real.stop()
        raise
    yield {"real": real, "empty": bare}
    real.stop()
    bare.stop()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(headless=True)
        except Exception as e:  # browser binaries not installed
            pytest.skip(f"Chromium for Playwright not available: {e}")
        yield b
        b.close()


def open_page(browser: Browser, url: str, w: int, h: int) -> tuple[Page, list[str]]:
    page = browser.new_page(viewport={"width": w, "height": h})
    problems: list[str] = []
    page.on("console", lambda m: problems.append(f"{m.type}: {m.text[:200]}") if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    page.goto(url)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(300)
    return page, problems


@pytest.mark.parametrize("w,h", WIDTHS, ids=[str(w) for w, _ in WIDTHS])
@pytest.mark.parametrize("pid", list(PAGES))
def test_page_renders(servers, browser, shots, pid, w, h):
    page, problems = open_page(browser, f"{servers['real'].url}/#{pid}", w, h)
    try:
        assert page.locator("h1").inner_text() == PAGES[pid][0]
        assert page.evaluate("document.documentElement.scrollWidth") <= w, "horizontal page scroll"
        assert page.locator("[aria-current=page]").inner_text() == page.locator(f'nav a[href="#{pid}"]').inner_text()
        assert page.get_by_role("alert").count() == 0, "API error banner shown"
        page.screenshot(path=str(shots / f"smoke_{pid}_{w}.png"), full_page=True)
        assert problems == [], problems
    finally:
        page.close()


@pytest.mark.parametrize("pid", list(PAGES))
def test_pending_states_without_results(servers, browser, shots, pid):
    page, problems = open_page(browser, f"{servers['empty'].url}/#{pid}", 1440, 1000)
    try:
        assert page.locator("h1").inner_text() == PAGES[pid][0]
        assert page.get_by_text(PAGES[pid][1]).first.is_visible()
        assert page.locator("table td").count() == 0 or pid == "elliptic", "data table rendered without results"
        page.screenshot(path=str(shots / f"smoke_empty_{pid}.png"), full_page=True)
        assert problems == [], problems
    finally:
        page.close()


@pytest.mark.parametrize("w,h", WIDTHS, ids=[str(w) for w, _ in WIDTHS])
def test_alerts_drawer_keyboard(servers, browser, shots, w, h):
    page, problems = open_page(browser, f"{servers['real'].url}/#alerts", w, h)
    try:
        btn = page.locator("tbody button").first
        row = btn.inner_text().replace("Open record", "").strip()
        btn.focus()
        page.keyboard.press("Enter")
        dialog = page.get_by_role("dialog")
        dialog.wait_for()
        assert dialog.locator("h2").inner_text() == f"Record {row}"
        assert page.evaluate("document.querySelector('[role=dialog]').contains(document.activeElement)")
        page.wait_for_load_state("networkidle")
        page.screenshot(path=str(shots / f"smoke_drawer_{w}.png"))
        page.keyboard.press("Escape")
        assert page.get_by_role("dialog").count() == 0
        assert page.evaluate("document.activeElement.innerText").replace("Open record", "").strip() == row
        page.keyboard.press(" ")
        page.get_by_role("dialog").wait_for()
        page.keyboard.press("Escape")
        assert problems == [], problems
    finally:
        page.close()


def test_mobile_nav_reaches_every_page(servers, browser):
    page, problems = open_page(browser, f"{servers['real'].url}/#overview", 390, 844)
    try:
        for pid, (h1, _) in PAGES.items():
            link = page.locator(f'nav a[href="#{pid}"]')
            box = link.bounding_box()
            assert box and box["x"] >= 0 and box["x"] + box["width"] <= 390, f"{pid} nav link off-screen"
            link.click()
            page.wait_for_function("t => document.querySelector('h1')?.innerText === t", arg=h1, timeout=10_000)
            assert page.url.endswith(f"#{pid}")
        assert problems == [], problems
    finally:
        page.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))


def test_strategies_spread_panel(servers, browser):
    """Separate-scenario panel: shown (labelled) when the linear_spread files exist, absent otherwise."""
    label = "Separate scenario: anomalies injected in E5–E10"
    have = any((ROOT / "results").glob("strategies_*_linear_spread.csv"))
    page, problems = open_page(browser, f"{servers['real'].url}/#strategies", 1440, 1000)
    try:
        page.wait_for_function("() => document.querySelectorAll('table').length > 0", timeout=10_000)
        page.wait_for_timeout(500)
        assert page.get_by_text(label).count() == (1 if have else 0)
        if have:
            assert page.get_by_role("heading", name="Anomaly-aware replay: what gets replayed when anomalies arrive earlier").is_visible()
        assert problems == [], problems
    finally:
        page.close()
    page, problems = open_page(browser, f"{servers['empty'].url}/#strategies", 1440, 1000)
    try:
        assert page.get_by_text(label).count() == 0
    finally:
        page.close()
