#!/usr/bin/env python3
"""GPU/CPU/메모리 실시간 모니터링 대시보드 서버.
1초 간격으로 샘플을 수집해 링버퍼에 쌓고, /api/stats(최신값)와
/api/history(최근 120초)로 내보낸다. 프론트엔드는 dashboard.html.
"""
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

PORT = 5757
HISTORY_LEN = 120
history = []
lock = threading.Lock()


def read_gpu():
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        if not out:
            return None
        util, temp, mem_used, mem_total, power_draw, power_limit = [
            float(x.strip()) for x in out.split(",")
        ]
        return {
            "util": util,
            "temp": temp,
            "mem_used": mem_used,
            "mem_total": mem_total,
            "power_draw": power_draw,
            "power_limit": power_limit,
        }
    except Exception:
        return None


def read_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        for key in ("coretemp", "k10temp", "zenpower"):
            entries = temps.get(key)
            if not entries:
                continue
            for entry in entries:
                if "Package" in (entry.label or "") or "Tctl" in (entry.label or ""):
                    return entry.current
            return entries[0].current
    except Exception:
        pass
    return None


def sample():
    cpu = psutil.cpu_percent(interval=None)
    per_cpu = psutil.cpu_percent(interval=None, percpu=True)
    mem = psutil.virtual_memory()
    return {
        "ts": time.time(),
        "cpu": {"percent": cpu, "per_cpu": per_cpu, "temp": read_cpu_temp()},
        "mem": {"percent": mem.percent, "used": mem.used, "total": mem.total},
        "gpu": read_gpu(),
    }


def collector():
    psutil.cpu_percent(interval=None)  # 첫 호출은 기준값이라 버린다
    while True:
        s = sample()
        with lock:
            history.append(s)
            if len(history) > HISTORY_LEN:
                del history[0]
        time.sleep(1)


HTML_PATH = Path(__file__).parent / "dashboard.html"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/stats":
            with lock:
                data = history[-1] if history else sample()
            self._send_json(data)
        elif self.path == "/api/history":
            with lock:
                data = list(history)
            self._send_json(data)
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    threading.Thread(target=collector, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"모니터링 대시보드: http://127.0.0.1:{PORT}")
    server.serve_forever()
