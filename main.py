from __future__ import annotations
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio, json, base64
from typing import Optional, Set, Dict, Any

import numpy as np
import cv2

app = FastAPI()

# -----------------------------
# Global state
# -----------------------------
rpi_connection: Optional[WebSocket] = None          # single Raspberry Pi/Ubuntu client
available_cameras: list[str] = []                   # from Pi "register"
frontend_connections: Set[WebSocket] = set()        # many admin browser clients
state_lock = asyncio.Lock()                         # serialize state mutations

# Track which cameras are actively streaming
active_cameras: Set[str] = set()

# Cache last result per camera: {"score": int, "image": base64_jpg}
last_results: Dict[str, Dict[str, Any]] = {}


# -----------------------------
# Dummy model (replace with real one later)
# -----------------------------
class DummyModel:
    def detect(self, image):
        return []  # demo: no detections
model = DummyModel()

def calculate_score(bullet_holes):
    score = 0
    for (x, y) in bullet_holes:
        score += 10
    return score


# -----------------------------
# Utilities
# -----------------------------
async def broadcast_json(payload: dict):
    """Send payload to all admin clients; remove any dead sockets."""
    dead = []
    for ws in list(frontend_connections):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            frontend_connections.remove(ws)
        except KeyError:
            pass

def build_snapshot() -> dict:
    """Full state snapshot for (re)connecting admin clients."""
    return {
        "action": "snapshot",
        "cameras": available_cameras,
        "active": list(active_cameras),
        "last_results": last_results,  # { cam_id: {score, image(b64)} }
    }


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def serve_frontend():
    """Serve index.html from the same directory."""
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.websocket("/ws/rpi")
async def ws_rpi_endpoint(websocket: WebSocket):
    """
    WebSocket for Raspberry Pi/Ubuntu client.

    Pi sends:
      - {"action":"register","cameras":[...]}
      - {"action":"frame","cam_id":"cam1","frame":"<base64 jpg>"}

    Server sends:
      - {"action":"start","cam_id":"cam1"}
      - {"action":"stop","cam_id":"cam1"}
      - {"action":"stop_all"}
    """
    global rpi_connection, available_cameras
    await websocket.accept()

    async with state_lock:
        rpi_connection = websocket

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "register":
                cams = data.get("cameras", [])
                if not isinstance(cams, list):
                    cams = []
                async with state_lock:
                    available_cameras = cams
                    active_cameras.intersection_update(cams)  # drop actives that disappeared
                await broadcast_json(build_snapshot())

            elif action == "frame":
                cam_id = data.get("cam_id")
                frame_b64 = data.get("frame")
                if not (cam_id and frame_b64):
                    continue

                try:
                    frame_bytes = base64.b64decode(frame_b64)
                    jpg_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    bullet_holes = model.detect(frame)
                    score = calculate_score(bullet_holes)

                    for (x, y) in bullet_holes:
                        cv2.circle(frame, (x, y), radius=15, color=(0, 0, 255), thickness=3)

                    ok, buffer = cv2.imencode(".jpg", frame)
                    if not ok:
                        continue
                    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

                    async with state_lock:
                        last_results[cam_id] = {"score": score, "image": annotated_b64}
                except Exception:
                    continue

                await broadcast_json({
                    "action": "result",
                    "cam_id": cam_id,
                    "score": score,
                    "image": annotated_b64
                })

            # ignore unknown actions

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        async with state_lock:
            if rpi_connection is websocket:
                rpi_connection = None
            available_cameras = []
            active_cameras.clear()
            last_results.clear()
        await broadcast_json(build_snapshot())


@app.websocket("/ws/client")
async def ws_client_endpoint(websocket: WebSocket):
    """WebSocket for admin browser clients."""
    await websocket.accept()
    async with state_lock:
        frontend_connections.add(websocket)

    # Send a full snapshot immediately (so refresh restores state)
    try:
        await websocket.send_json(build_snapshot())
    except Exception:
        async with state_lock:
            frontend_connections.discard(websocket)
        return

    try:
        while True:
            # keepalive; updates are pushed from elsewhere
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        async with state_lock:
            frontend_connections.discard(websocket)


@app.get("/start/{cam_id}")
async def start_camera(cam_id: str):
    """Tell the Pi to start streaming a given camera."""
    async with state_lock:
        ws = rpi_connection
        cams = set(available_cameras)
        already_active = cam_id in active_cameras
    if ws is None:
        return {"status": "error", "detail": "Raspberry Pi not connected"}
    if cam_id not in cams:
        return {"status": "error", "detail": f"Camera {cam_id} not available"}

    # Guard duplicate start
    if already_active:
        return {"status": "ok", "detail": f"{cam_id} already running"}

    try:
        await ws.send_json({"action": "start", "cam_id": cam_id})
        async with state_lock:
            active_cameras.add(cam_id)
        await broadcast_json(build_snapshot())
        return {"status": "ok", "detail": f"Start command sent for {cam_id}"}
    except Exception as e:
        return {"status": "error", "detail": f"Failed to send start: {e}"}


@app.get("/stop/{cam_id}")
async def stop_camera(cam_id: str):
    """Tell the Pi to stop streaming a given camera (keep last image/score cached)."""
    async with state_lock:
        ws = rpi_connection
        is_active = cam_id in active_cameras
    if ws is None:
        return {"status": "error", "detail": "Raspberry Pi not connected"}

    # Guard duplicate stop
    if not is_active:
        return {"status": "ok", "detail": f"{cam_id} already stopped"}

    try:
        await ws.send_json({"action": "stop", "cam_id": cam_id})
        async with state_lock:
            active_cameras.discard(cam_id)
        await broadcast_json(build_snapshot())
        return {"status": "ok", "detail": f"Stop command sent for {cam_id}"}
    except Exception as e:
        return {"status": "error", "detail": f"Failed to send stop: {e}"}


@app.get("/stop_all")
async def stop_all():
    async with state_lock:
        ws = rpi_connection
    if ws is None:
        return {"status": "error", "detail": "Raspberry Pi not connected"}
    try:
        await ws.send_json({"action": "stop_all"})
        async with state_lock:
            active_cameras.clear()
        await broadcast_json(build_snapshot())
        return {"status": "ok", "detail": "Stop all command sent"}
    except Exception as e:
        return {"status": "error", "detail": f"Failed to send stop_all: {e}"}
