#!/usr/bin/env python3
"""
Version 1 — Live-App & Mobile/Tablet PWA Server (Mac + iPad + iPhone + Bruder-App)
Serves:
  - GET  /                         -> Version_1_App.html (Full PWA Dashboard + 4Y Backtest)
  - GET  /manifest.json            -> Web App Manifest for iOS/iPadOS/Android Home Screen
  - GET  /apple-touch-icon.png     -> 180x180 High-Res V1 App Icon
  - GET  /icon-512.png             -> 512x512 High-Res V1 App Icon
  - GET  /download/Version_1_App.html -> Direct file download for sharing with brother
  - GET  /api/status               -> Returns LAN IP URL & optional Public Tunnel URL
  - POST /api/refresh              -> Runs live EODHD + Yahoo data scan & rebuilds Version_1_App.html
  - POST /api/start_tunnel         -> Starts an instant public HTTPS tunnel (localhost.run) for brother access anywhere
"""

import http.server
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
PORT = int(os.environ.get("PORT", os.environ.get("V1_PORT", "8765")))

from datetime import datetime

TUNNEL_PROC = None
PUBLIC_URL = os.environ.get("RENDER_EXTERNAL_URL") or "https://version-1-umwa.onrender.com"
IS_RENDER_CLOUD = bool(os.environ.get("RENDER") or os.environ.get("RENDER_EXTERNAL_URL"))
REFRESH_LOCK = threading.Lock()
CLOUD_SYNC_LOCK = threading.Lock()
POSITIONS_LOCK = threading.Lock()
POSITIONS_FILE = BASE_DIR / "positions.json"
IS_REFRESHING = False
LAST_REFRESH_OK = True
LAST_REFRESH_TS = int(time.time())
LAST_DURATION_SEC = 0.0
LAST_CLOUD_SYNC_TS = 0
REFRESH_START_TS = 0


import agent

def _warmup_in_memory_cache():
    """Lädt beim Server-Start sofort den Kerzen-Cache in den RAM und wärmt die Indikator-Engine vor."""
    try:
        agent._load_candle_cache()
        if agent._CANDLE_CACHE_MEM:
            with agent.ThreadPoolExecutor(max_workers=8 if IS_RENDER_CLOUD else 24) as pool:
                list(pool.map(lambda s: agent._process_one(s, "d", None), agent.DEFAULT_WATCHLIST))
    except Exception:
        pass


def _sync_to_cloud_worker():
    """
    Hybrid-Turbo (Option 1):
    Pusht das auf dem M1 Mac mini fertig berechnete Dashboard im Hintergrund
    direkt zu trading-landing-page (Branch main -> https://www.market-vision-pro.com).
    """
    global LAST_CLOUD_SYNC_TS
    if IS_RENDER_CLOUD:
        return
    with CLOUD_SYNC_LOCK:
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        src_html = BASE_DIR / "Version_1_App.html"
        try:
            landing_dir = BASE_DIR.parent / "trading-landing-page"
            if landing_dir.exists() and (landing_dir / ".git").exists() and src_html.exists():
                html_bytes = src_html.read_bytes()
                (landing_dir / "index.html").write_bytes(html_bytes)
                (landing_dir / "Version_1_App.html").write_bytes(html_bytes)
                subprocess.run(
                    ["git", "add", "index.html", "Version_1_App.html", "manifest.json", "apple-touch-icon.png", "icon-512.png"],
                    cwd=str(landing_dir),
                    capture_output=True,
                    timeout=20,
                )
                diff_res2 = subprocess.run(
                    ["git", "diff", "--cached", "--quiet"],
                    cwd=str(landing_dir),
                    capture_output=True,
                    timeout=10,
                )
                if diff_res2.returncode != 0:
                    subprocess.run(
                        ["git", "commit", "-m", f"Auto-Sync Mac mini M1 Scan to market-vision-pro.com ({ts_str})"],
                        cwd=str(landing_dir),
                        capture_output=True,
                        timeout=20,
                    )
                push_res = subprocess.run(
                    ["git", "push", "origin", "main"],
                    cwd=str(landing_dir),
                    capture_output=True,
                    timeout=45,
                )
                if push_res.returncode == 0:
                    LAST_CLOUD_SYNC_TS = int(time.time())
        except Exception:
            pass


def _run_refresh_worker():
    global IS_REFRESHING, LAST_REFRESH_OK, LAST_REFRESH_TS, REFRESH_START_TS, LAST_DURATION_SEC
    with REFRESH_LOCK:
        IS_REFRESHING = True
        t0 = time.time()
        REFRESH_START_TS = int(t0)
        try:
            out = agent.scan_symbols(agent.DEFAULT_WATCHLIST, period="d", only_signals=True)
            src_html = BASE_DIR / "Version_1_App.html"
            dash_html = BASE_DIR / "Stillhalter_Dashboard.html"
            agent.export_html_dashboard(out, str(dash_html))
            desktop_html = Path.home() / "Desktop" / "Version_1_App.html"
            if src_html.exists() and desktop_html.parent.exists():
                try:
                    desktop_html.write_bytes(src_html.read_bytes())
                except Exception:
                    pass
            LAST_REFRESH_OK = True
            LAST_REFRESH_TS = int(time.time())
            LAST_DURATION_SEC = round(time.time() - t0, 2)
            if not IS_RENDER_CLOUD:
                threading.Thread(target=_sync_to_cloud_worker, daemon=True).start()
        except Exception:
            LAST_REFRESH_OK = False
        finally:
            IS_REFRESHING = False


def _market_scheduler_loop():
    """
    24/7 Hintergrund-Planer auf dem Mac mini:
    Scannt an Handelstagen (Mo–Fr) automatisch zu den Kernzeiten (09:00, 15:35 US-Open,
    17:30, 20:00, 22:05 US-Close) und synchronisiert das Ergebnis zu Render/GitHub.
    """
    if IS_RENDER_CLOUD:
        return
    schedule_slots = {"09:00", "15:35", "17:30", "20:00", "22:05"}
    triggered_key = ""
    while True:
        try:
            now = datetime.now()
            if now.weekday() < 5:  # Mo (0) bis Fr (4)
                hm = now.strftime("%H:%M")
                slot_key = f"{now.strftime('%Y-%m-%d')}_{hm}"
                if hm in schedule_slots and slot_key != triggered_key:
                    triggered_key = slot_key
                    trigger_async_refresh()
        except Exception:
            pass
        time.sleep(30)


def trigger_async_refresh() -> bool:
    if IS_REFRESHING:
        return True
    t = threading.Thread(target=_run_refresh_worker, daemon=True)
    t.start()
    return True


def get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def start_public_tunnel() -> str:
    global TUNNEL_PROC, PUBLIC_URL
    if PUBLIC_URL and TUNNEL_PROC and TUNNEL_PROC.poll() is None:
        return PUBLIC_URL

    try:
        if TUNNEL_PROC and TUNNEL_PROC.poll() is None:
            TUNNEL_PROC.terminate()
    except Exception:
        pass

    PUBLIC_URL = None
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ServerAliveInterval=30",
        "-R", f"80:localhost:{PORT}",
        "nokey@localhost.run",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        TUNNEL_PROC = proc

        def reader():
            global PUBLIC_URL
            if not proc.stdout:
                return
            for line in proc.stdout:
                m = re.search(r"(https://[a-zA-Z0-9.-]+\.lhr\.life)", line)
                if m:
                    PUBLIC_URL = m.group(1)

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        # Wait up to 6 seconds for URL
        for _ in range(30):
            if PUBLIC_URL:
                return PUBLIC_URL
            time.sleep(0.2)
    except Exception:
        pass
    return PUBLIC_URL or ""


class Version1Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Keep background daemon clean

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html", "/Version_1_App.html", "/Stillhalter_Dashboard.html"):
            html_file = BASE_DIR / "Version_1_App.html"
            if not html_file.exists():
                html_file = BASE_DIR / "Stillhalter_Dashboard.html"
            if html_file.exists():
                data = html_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404, "Dashboard HTML not found")
            return

        if path == "/download/Version_1_App.html":
            html_file = BASE_DIR / "Version_1_App.html"
            if not html_file.exists():
                html_file = BASE_DIR / "Stillhalter_Dashboard.html"
            if html_file.exists():
                data = html_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="Version_1_App.html"')
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404, "File not found")
            return

        if path == "/manifest.json":
            manifest = {
                "name": "Version 1 — UMWA Quantitative Option Desk",
                "short_name": "Version 1",
                "description": "UMWA 4/4 Stillhalter Scanner, Elliott Wave, Fundamental Matrix & 4J Short-Put Backtest",
                "start_url": "/",
                "display": "standalone",
                "background_color": "#07090e",
                "theme_color": "#07090e",
                "orientation": "any",
                "icons": [
                    {"src": "/apple-touch-icon.png", "sizes": "180x180", "type": "image/png", "purpose": "any maskable"},
                    {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
                ],
            }
            payload = json.dumps(manifest, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/manifest+json; charset=utf-8")
            self._send_cors()
            self.end_headers()
            self.wfile.write(payload)
            return

        if path in ("/apple-touch-icon.png", "/apple-touch-icon-precomposed.png", "/favicon.ico"):
            icon_path = ASSETS_DIR / "apple_touch_icon.png"
            if icon_path.exists():
                data = icon_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)
            return

        if path == "/icon-512.png":
            icon_path = ASSETS_DIR / "icon_512.png"
            if icon_path.exists():
                data = icon_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)
            return

        if path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"status":"ok","version":"Version 1"}')
            return

        if path == "/api/positions":
            positions = []
            with POSITIONS_LOCK:
                if POSITIONS_FILE.exists():
                    try:
                        raw = json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))
                        if isinstance(raw, list):
                            positions = raw
                        elif isinstance(raw, dict) and isinstance(raw.get("positions"), list):
                            positions = raw["positions"]
                    except Exception:
                        positions = []
            payload = json.dumps({"ok": True, "positions": positions}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors()
            self.end_headers()
            self.wfile.write(payload)
            return

        if path == "/api/status":
            lan_ip = get_lan_ip()
            ext_url = os.environ.get("RENDER_EXTERNAL_URL") or PUBLIC_URL
            elapsed = (int(time.time()) - REFRESH_START_TS) if IS_REFRESHING and REFRESH_START_TS else 0
            resp = {
                "ok": True,
                "version": "Version 1",
                "host": "render-cloud" if IS_RENDER_CLOUD else "mac-mini-m1",
                "lan_url": f"http://{lan_ip}:{PORT}",
                "public_url": ext_url,
                "refreshing": IS_REFRESHING,
                "elapsed_sec": elapsed,
                "last_refresh_ok": LAST_REFRESH_OK,
                "last_refresh_ts": LAST_REFRESH_TS,
                "last_duration_sec": LAST_DURATION_SEC,
                "last_cloud_sync_ts": LAST_CLOUD_SYNC_TS,
            }
            payload = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors()
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        path = self.path.split("?")[0]

        if path == "/api/positions":
            try:
                clen = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(clen).decode("utf-8") if clen > 0 else "{}"
                data = json.loads(body)
                positions = data.get("positions", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                if not isinstance(positions, list):
                    positions = []
                with POSITIONS_LOCK:
                    POSITIONS_FILE.write_text(json.dumps({"positions": positions, "updated_ts": int(time.time())}, indent=2), encoding="utf-8")
                payload = json.dumps({"ok": True, "count": len(positions)}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._send_cors()
                self.end_headers()
                self.wfile.write(payload)
            except Exception as e:
                payload = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._send_cors()
                self.end_headers()
                self.wfile.write(payload)
            return

        if path == "/api/refresh":
            trigger_async_refresh()
            payload = json.dumps({
                "ok": True,
                "refreshing": True,
                "last_refresh_ts": LAST_REFRESH_TS,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors()
            self.end_headers()
            self.wfile.write(payload)
            return

        if path == "/api/start_tunnel":
            url = start_public_tunnel()
            payload = json.dumps({"ok": bool(url), "public_url": url}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors()
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_error(404, "Not Found")


class ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True


def main():
    threading.Thread(target=_warmup_in_memory_cache, daemon=True).start()
    if not IS_RENDER_CLOUD:
        threading.Thread(target=_market_scheduler_loop, daemon=True).start()
    lan_ip = get_lan_ip()
    server = ThreadedHTTPServer(("0.0.0.0", PORT), Version1Handler)
    print(f"Version 1 Live-Server aktiv auf:")
    print(f"  Mac mini (Lokal): http://localhost:{PORT}")
    print(f"  iPad / Handy LAN: http://{lan_ip}:{PORT}")
    print(f"  Global Cloud Sync: {PUBLIC_URL}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
