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

TUNNEL_PROC = None
PUBLIC_URL = os.environ.get("RENDER_EXTERNAL_URL") or "https://version-1-umwa.onrender.com"
REFRESH_LOCK = threading.Lock()
IS_REFRESHING = False
LAST_REFRESH_OK = True


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

        if path == "/api/status":
            lan_ip = get_lan_ip()
            ext_url = os.environ.get("RENDER_EXTERNAL_URL") or PUBLIC_URL
            resp = {
                "ok": True,
                "version": "Version 1",
                "lan_url": f"http://{lan_ip}:{PORT}",
                "public_url": ext_url,
                "refreshing": IS_REFRESHING,
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

        if path == "/api/refresh":
            with REFRESH_LOCK:
                try:
                    python_bin = BASE_DIR / "venv" / "bin" / "python"
                    if not python_bin.exists():
                        python_bin = Path(sys.executable)
                    res = subprocess.run(
                        [str(python_bin), str(BASE_DIR / "agent.py"), "--only-signals", "--html"],
                        cwd=str(BASE_DIR),
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    # Also copy updated Version_1_App.html to Desktop if Desktop exists
                    desktop_html = Path.home() / "Desktop" / "Version_1_App.html"
                    src_html = BASE_DIR / "Version_1_App.html"
                    if src_html.exists() and desktop_html.parent.exists():
                        try:
                            desktop_html.write_bytes(src_html.read_bytes())
                        except Exception:
                            pass
                    ok = res.returncode == 0
                    payload = json.dumps({"ok": ok, "returncode": res.returncode}).encode("utf-8")
                except Exception as e:
                    payload = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")

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
    lan_ip = get_lan_ip()
    server = ThreadedHTTPServer(("0.0.0.0", PORT), Version1Handler)
    print(f"Version 1 Live-Server aktiv auf:")
    print(f"  MacBook:      http://localhost:{PORT}")
    print(f"  iPad / Handy: http://{lan_ip}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
