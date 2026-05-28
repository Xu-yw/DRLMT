#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA 训练画面观察器。

这个脚本是独立调试工具：它只连接当前正在运行的 CARLA world，
找到训练车辆，临时挂载一个 RGB 相机，用 pygame 把相机帧转成画布，
再通过本地 HTTP 端口输出连续图片流。脚本不会调用 load_world()，
不会修改 PPO 训练循环，也不会接触 mutation hook。
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# tools/ 下运行时，手动把仓库根目录加入 import path。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CARLA_EGG_DIR = PROJECT_ROOT / "carla"


def _prepare_imports() -> object:
    """加载 CARLA egg，并返回 carla 模块。"""
    py_tag = f"{sys.version_info.major}.{sys.version_info.minor}"
    eggs = sorted(CARLA_EGG_DIR.glob(f"carla-*{py_tag}-linux-x86_64.egg"))
    if eggs:
        sys.path.insert(0, str(eggs[0]))
    else:
        # 保底走项目已有的 connection.py，方便沿用历史环境配置。
        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            from simulation.connection import carla as project_carla
            return project_carla
        except Exception:
            pass

    import carla  # type: ignore

    return carla


def _prepare_pygame(show_window: bool) -> object:
    """初始化 pygame。无 DISPLAY 时默认走 dummy driver，便于远程网页流运行。"""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if not show_window and not os.environ.get("DISPLAY"):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame  # pylint: disable=import-outside-toplevel

    pygame.init()
    return pygame


carla = _prepare_imports()


class FrameStore:
    """线程安全地保存最新一帧，供 HTTP handler 读取。"""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame_id = 0
        self._payload: Optional[bytes] = None
        self._content_type = "image/jpeg"
        self._carla_frame = -1
        self._updated_at = 0.0
        self._vehicle_id: Optional[int] = None

    def update(self, payload: bytes, content_type: str, carla_frame: int, vehicle_id: Optional[int]) -> None:
        with self._condition:
            self._frame_id += 1
            self._payload = payload
            self._content_type = content_type
            self._carla_frame = carla_frame
            self._updated_at = time.time()
            self._vehicle_id = vehicle_id
            self._condition.notify_all()

    def wait_for_newer(self, last_seen: int, timeout: float) -> Optional[Tuple[int, bytes, str, int, float, Optional[int]]]:
        deadline = time.time() + timeout
        with self._condition:
            while self._payload is None or self._frame_id <= last_seen:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return (
                self._frame_id,
                self._payload,
                self._content_type,
                self._carla_frame,
                self._updated_at,
                self._vehicle_id,
            )

    def latest(self, timeout: float = 0.0) -> Optional[Tuple[int, bytes, str, int, float, Optional[int]]]:
        return self.wait_for_newer(0, timeout) if self._payload is None else self.wait_for_newer(-1, 0.0)


class PygameEncoder:
    """用 pygame surface 编码图片；JPEG 不可用时自动降级为 PNG。"""

    def __init__(self, pygame_module: object, image_format: str = "jpeg") -> None:
        self.pygame = pygame_module
        self.image_format = image_format
        self._tmp_dir = Path("/dev/shm") if Path("/dev/shm").exists() else Path("/tmp")
        self._tmp_path = self._tmp_dir / f"carla_pygame_observer_{os.getpid()}.jpg"
        self._warned_fallback = False

    def encode(self, surface: object) -> Tuple[bytes, str]:
        if self.image_format == "jpeg":
            try:
                self._tmp_path = self._tmp_path.with_suffix(".jpg")
                self.pygame.image.save(surface, str(self._tmp_path))
                return self._tmp_path.read_bytes(), "image/jpeg"
            except Exception as exc:  # pygame 在部分构建里没有 JPEG save 支持。
                if not self._warned_fallback:
                    print(f"[observer] JPEG encode failed, fallback to PNG: {exc}", flush=True)
                    self._warned_fallback = True
                self.image_format = "png"

        self._tmp_path = self._tmp_path.with_suffix(".png")
        self.pygame.image.save(surface, str(self._tmp_path))
        return self._tmp_path.read_bytes(), "image/png"

    def cleanup(self) -> None:
        for suffix in (".jpg", ".png"):
            path = self._tmp_path.with_suffix(suffix)
            try:
                path.unlink()
            except FileNotFoundError:
                pass


class CarlaPygameObserver:
    """负责发现训练车辆、挂载相机、接收相机帧。"""

    def __init__(self, args: argparse.Namespace, store: FrameStore, pygame_module: object) -> None:
        self.args = args
        self.store = store
        self.pygame = pygame_module
        self.encoder = PygameEncoder(pygame_module, args.image_format)
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(args.timeout)
        # 关键：只取当前 world，不 load_world，否则会重置训练环境。
        self.world = self.client.get_world()
        self.sensor = None
        self.vehicle = None
        self.display = None
        self._last_encode_at = 0.0
        self._frame_interval = 1.0 / max(args.max_fps, 1.0)
        self._lock = threading.Lock()

        if args.show_window:
            self.display = self.pygame.display.set_mode((args.width, args.height))
            self.pygame.display.set_caption("CARLA pygame observer")

    def find_vehicle(self):
        vehicles = list(self.world.get_actors().filter(self.args.vehicle_filter))
        if not vehicles:
            return None

        preferred = []
        fallback = []
        for vehicle in vehicles:
            type_id = getattr(vehicle, "type_id", "")
            role_name = vehicle.attributes.get("role_name", "") if hasattr(vehicle, "attributes") else ""
            try:
                velocity = vehicle.get_velocity()
                speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
            except Exception:
                speed = 0.0
            item = (role_name in {"hero", "ego", "ego_vehicle"}, type_id == self.args.prefer_type, speed, vehicle.id, vehicle)
            if type_id == self.args.prefer_type:
                preferred.append(item)
            else:
                fallback.append(item)

        candidates = preferred or fallback
        candidates.sort(reverse=True)
        return candidates[0][-1]

    def attach_if_needed(self) -> bool:
        # 训练环境每个 episode 会销毁并重建车辆，所以这里每秒重新确认当前车辆。
        vehicle = self.find_vehicle()
        if vehicle is None:
            self.destroy_sensor()
            print("[observer] no vehicle found; waiting...", flush=True)
            return False

        if self._vehicle_alive(vehicle) and self._sensor_alive(vehicle):
            return True

        self.destroy_sensor()

        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.args.width))
        camera_bp.set_attribute("image_size_y", str(self.args.height))
        camera_bp.set_attribute("fov", str(self.args.fov))
        if camera_bp.has_attribute("sensor_tick"):
            camera_bp.set_attribute("sensor_tick", f"{self._frame_interval:.4f}")

        transform = carla.Transform(
            carla.Location(x=self.args.camera_x, y=self.args.camera_y, z=self.args.camera_z),
            carla.Rotation(pitch=self.args.camera_pitch, yaw=self.args.camera_yaw, roll=0.0),
        )

        try:
            sensor = self.world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        except Exception as exc:
            print(f"[observer] failed to spawn camera on vehicle {vehicle.id}: {exc}", flush=True)
            return False

        self.vehicle = vehicle
        self.sensor = sensor
        weak_self = self
        sensor.listen(lambda image: weak_self._on_image(image))
        print(f"[observer] attached camera {sensor.id} to vehicle {vehicle.id} ({vehicle.type_id})", flush=True)
        return True

    def _vehicle_alive(self, vehicle=None) -> bool:
        vehicle = vehicle or self.vehicle
        if vehicle is None:
            return False
        try:
            current = self.world.get_actor(vehicle.id)
            return bool(current is not None and current.is_alive and current.type_id.startswith("vehicle."))
        except Exception:
            return False

    def _sensor_alive(self, vehicle=None) -> bool:
        if self.sensor is None:
            return False
        try:
            current = self.world.get_actor(self.sensor.id)
            if current is None or not current.is_alive:
                return False
            if vehicle is None:
                return True
            parent = current.parent
            return bool(parent is not None and parent.id == vehicle.id)
        except Exception:
            return False

    def _on_image(self, image) -> None:
        now = time.monotonic()
        if now - self._last_encode_at < self._frame_interval:
            return
        self._last_encode_at = now

        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            rgb = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])
            surface = self.pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

            if self.display is not None:
                with self._lock:
                    self.display.blit(surface, (0, 0))
                    self.pygame.display.flip()
                    self.pygame.event.pump()

            payload, content_type = self.encoder.encode(surface)
            vehicle_id = self.vehicle.id if self.vehicle is not None else None
            self.store.update(payload, content_type, int(image.frame), vehicle_id)
        except Exception as exc:
            print(f"[observer] frame callback failed: {exc}", flush=True)

    def destroy_sensor(self) -> None:
        sensor = self.sensor
        self.sensor = None
        if sensor is None:
            return
        try:
            sensor.stop()
        except Exception:
            pass
        try:
            if sensor.is_alive:
                sensor.destroy()
                print(f"[observer] destroyed camera {sensor.id}", flush=True)
        except Exception as exc:
            print(f"[observer] failed to destroy camera: {exc}", flush=True)

    def close(self) -> None:
        self.destroy_sensor()
        self.encoder.cleanup()
        try:
            self.pygame.quit()
        except Exception:
            pass


class ObserverHttpHandler(BaseHTTPRequestHandler):
    server_version = "CarlaPygameObserver/1.0"

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"[http] {self.address_string()} - {fmt % args}", flush=True)

    @property
    def frame_store(self) -> FrameStore:
        return self.server.frame_store  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._send_index()
        elif self.path == "/snapshot.jpg":
            self._send_snapshot()
        elif self.path == "/stream.mjpg":
            self._send_stream()
        elif self.path == "/healthz":
            self._send_health()
        else:
            self.send_error(404, "not found")

    def _send_index(self) -> None:
        html = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>CARLA pygame observer</title>
<style>
body{margin:0;background:#101418;color:#e8eef5;font:16px/1.5 system-ui,Segoe UI,sans-serif;}
main{max-width:960px;margin:28px auto;padding:0 20px;}
img{width:100%;height:auto;background:#05070a;border:1px solid #2b3440;}
code{color:#9bd3ff;}
</style>
</head>
<body>
<main>
<h1>CARLA pygame observer</h1>
<p>远程训练画面流。若画面为空，说明当前还未找到训练车辆或相机帧尚未到达。</p>
<img src="/stream.mjpg" alt="CARLA stream">
<p>单帧地址：<code>/snapshot.jpg</code>；健康检查：<code>/healthz</code></p>
</main>
</body>
</html>
"""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_snapshot(self) -> None:
        frame = self.frame_store.latest(timeout=10.0)
        if frame is None:
            self.send_error(503, "no frame available")
            return
        _, payload, content_type, carla_frame, updated_at, vehicle_id = frame
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Carla-Frame", str(carla_frame))
        self.send_header("X-Updated-At", f"{updated_at:.3f}")
        if vehicle_id is not None:
            self.send_header("X-Vehicle-Id", str(vehicle_id))
        self.end_headers()
        self.wfile.write(payload)

    def _send_stream(self) -> None:
        boundary = "frame"
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.end_headers()

        last_seen = 0
        while True:
            frame = self.frame_store.wait_for_newer(last_seen, timeout=10.0)
            if frame is None:
                continue
            frame_id, payload, content_type, carla_frame, updated_at, vehicle_id = frame
            last_seen = frame_id
            headers = [
                f"--{boundary}",
                f"Content-Type: {content_type}",
                f"Content-Length: {len(payload)}",
                f"X-Observer-Frame: {frame_id}",
                f"X-Carla-Frame: {carla_frame}",
                f"X-Updated-At: {updated_at:.3f}",
            ]
            if vehicle_id is not None:
                headers.append(f"X-Vehicle-Id: {vehicle_id}")
            try:
                self.wfile.write(("\r\n".join(headers) + "\r\n\r\n").encode("ascii"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

    def _send_health(self) -> None:
        frame = self.frame_store.latest(timeout=0.0)
        body = b"ok\n" if frame is not None else b"waiting_for_frame\n"
        self.send_response(200 if frame is not None else 503)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CARLA pygame observer with browser stream output")
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port")
    parser.add_argument("--timeout", type=float, default=10.0, help="CARLA client timeout seconds")
    parser.add_argument("--web-host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--web-port", type=int, default=8090, help="HTTP bind port")
    parser.add_argument("--width", type=int, default=720, help="camera width")
    parser.add_argument("--height", type=int, default=720, help="camera height")
    parser.add_argument("--fov", type=float, default=90.0, help="camera field of view")
    parser.add_argument("--max-fps", type=float, default=10.0, help="max encoded stream FPS")
    parser.add_argument("--image-format", choices=("jpeg", "png"), default="jpeg", help="preferred stream encoding")
    parser.add_argument("--vehicle-filter", default="vehicle.*", help="CARLA actor filter for vehicles")
    parser.add_argument("--prefer-type", default="vehicle.tesla.model3", help="preferred ego vehicle type_id")
    parser.add_argument("--camera-x", type=float, default=-4.0, help="relative camera x")
    parser.add_argument("--camera-y", type=float, default=0.0, help="relative camera y")
    parser.add_argument("--camera-z", type=float, default=2.0, help="relative camera z")
    parser.add_argument("--camera-pitch", type=float, default=-12.0, help="relative camera pitch")
    parser.add_argument("--camera-yaw", type=float, default=0.0, help="relative camera yaw")
    parser.add_argument("--show-window", action="store_true", help="also open a pygame window when DISPLAY/VNC is available")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pygame_module = _prepare_pygame(args.show_window)
    store = FrameStore()
    observer = CarlaPygameObserver(args, store, pygame_module)
    stop_event = threading.Event()

    def _stop(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    httpd = ThreadingHTTPServer((args.web_host, args.web_port), ObserverHttpHandler)
    httpd.frame_store = store  # type: ignore[attr-defined]
    http_thread = threading.Thread(target=httpd.serve_forever, name="observer-http", daemon=True)
    http_thread.start()
    print(f"[observer] web stream: http://{args.web_host}:{args.web_port}/", flush=True)
    print(f"[observer] connected to CARLA {args.host}:{args.port}; world={observer.world.get_map().name}", flush=True)

    try:
        while not stop_event.is_set():
            observer.attach_if_needed()
            time.sleep(1.0)
    finally:
        httpd.shutdown()
        httpd.server_close()
        observer.close()
        print("[observer] stopped", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
