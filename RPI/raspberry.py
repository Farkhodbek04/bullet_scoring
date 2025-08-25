import asyncio
import json
import base64
import signal
from pathlib import Path

import cv2
import websockets

# >>>> Match your backend exactly (include :8000 if uvicorn runs on 8000) <<<<
SERVER_WS_URL = "ws://0.0.0.0:8000/ws/rpi"

# Tuning
FPS = 20
JPEG_QUALITY = 30
RECONNECT_DELAY_SEC = 0.001

CAMERAS_JSON = Path(__file__).with_name("cameras.json")  # same folder as script


def open_capture_from_source(source):
    """
    Accept int (e.g., 0), numeric str ("0"), or URL (rtsp/http/file).
    On Ubuntu/RPi, prefer V4L2 for local devices.
    """
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if isinstance(source, int):
        return cv2.VideoCapture(source, cv2.CAP_V4L2)

    return cv2.VideoCapture(source)


async def stream_camera(cam_id: str, source, ws):
    """Capture frames and send to server until cancelled."""
    cap = open_capture_from_source(source)
    if not cap.isOpened():
        print(f"[{cam_id}] ERROR: Failed to open source: {source}")
        return

    try:
        print(f"[{cam_id}] Streaming started")
        period = 1.0 / max(FPS, 1)
        while True:
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.2)
                continue

            ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                continue

            b64 = base64.b64encode(enc).decode("utf-8")
            await ws.send(json.dumps({"action": "frame", "cam_id": cam_id, "frame": b64}))
            await asyncio.sleep(period)

    except asyncio.CancelledError:
        print(f"[{cam_id}] Streaming cancelled")
    finally:
        cap.release()
        print(f"[{cam_id}] Streaming stopped")


async def run_client():
    if not CAMERAS_JSON.exists():
        print(f"ERROR: {CAMERAS_JSON} not found. Create it like: {{\"cam1\":\"rtsp://...\",\"cam2\":0}}")
        return

    with CAMERAS_JSON.open("r", encoding="utf-8") as f:
        camera_map = json.load(f)  # values can be int or str

    tasks: dict[str, asyncio.Task] = {}

    while True:
        try:
            print(f"Connecting to backend: {SERVER_WS_URL}")
            async with websockets.connect(
                SERVER_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                max_size=20_000_000,
            ) as ws:
                print("Connected. Registering cameras:", list(camera_map.keys()))
                await ws.send(json.dumps({"action": "register", "cameras": list(camera_map.keys())}))

                while True:
                    msg_text = await ws.recv()
                    data = json.loads(msg_text)
                    action = data.get("action")

                    if action == "start":
                        cam_id = data.get("cam_id")
                        if cam_id in camera_map and cam_id not in tasks:
                            print(f"START {cam_id}")
                            tasks[cam_id] = asyncio.create_task(
                                stream_camera(cam_id, camera_map[cam_id], ws)
                            )
                        else:
                            print(f"START ignored for {cam_id} (not found or already running)")

                    elif action == "stop":
                        cam_id = data.get("cam_id")
                        if cam_id in tasks:
                            print(f"STOP {cam_id}")
                            tasks[cam_id].cancel()
                            try:
                                await tasks[cam_id]
                            except asyncio.CancelledError:
                                pass
                            del tasks[cam_id]
                        else:
                            print(f"STOP ignored for {cam_id} (not running)")

                    elif action == "stop_all":
                        print("STOP ALL")
                        for cid, t in list(tasks.items()):
                            t.cancel()
                            try:
                                await t
                            except asyncio.CancelledError:
                                pass
                            del tasks[cid]

        except (websockets.ConnectionClosed, OSError) as e:
            print(f"WS disconnected: {e}. Reconnecting in {RECONNECT_DELAY_SEC}s...")
        except Exception as e:
            print(f"WS error: {e}. Reconnecting in {RECONNECT_DELAY_SEC}s...")

        # Ensure all streams stop on disconnect
        for cid, t in list(tasks.items()):
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            del tasks[cid]

        await asyncio.sleep(RECONNECT_DELAY_SEC)


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    stop_event = asyncio.Event()

    def handle_sig(*_):
        loop.create_task(stop_event.set())

    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, handle_sig)

    async def runner():
        client_task = asyncio.create_task(run_client())
        await stop_event.wait()
        client_task.cancel()
        try:
            await client_task
        except asyncio.CancelledError:
            pass

    try:
        loop.run_until_complete(runner())
    finally:
        loop.close()
        print("Client exited cleanly")


if __name__ == "__main__":
    # Ubuntu tip: ensure your user can access /dev/video0
    #   sudo usermod -a -G video $USER && newgrp video
    main()
