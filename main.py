from __future__ import annotations
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio, json, base64
from typing import Optional, Set, Dict, Any

import numpy as np
import cv2

from model import BulletHoleModel  # YOLO + ring scoring

app = FastAPI()

# -----------------------------
# Global state
# -----------------------------
rpi_connection: Optional[WebSocket] = None
available_cameras: list[str] = []
frontend_connections: Set[WebSocket] = set()
state_lock = asyncio.Lock()

# Which cameras are actively streaming
active_cameras: Set[str] = set()

# Last result per camera: {"score": int, "image": base64_jpg, "points": [...]}
last_results: Dict[str, Dict[str, Any]] = {}

# Baseline handling per camera (state before soldier shoots)
baseline_pending: Dict[str, bool] = {}   # armed by /start -> next processed frame becomes baseline
baseline_score: Dict[str, int] = {}
baseline_points: Dict[str, Any] = {}

# Per-camera frame queues and processing workers
frame_queues: Dict[str, asyncio.Queue] = {}   # cam_id -> Queue[str base64]
worker_tasks: Dict[str, asyncio.Task] = {}    # cam_id -> Task

# -----------------------------
# Load your trained model once
# -----------------------------
model = BulletHoleModel(weights_path="best.pt", conf_thres=0.25)


# -----------------------------
# Utilities
# -----------------------------
async def broadcast_json(payload: dict):
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
    return {
        "action": "snapshot",
        "cameras": available_cameras,
        "active": list(active_cameras),
        "last_results": last_results,  # { cam_id: { score, image(b64), points:[{x,y,conf,ring,score}] } }
    }

def ensure_queue(cam_id: str) -> asyncio.Queue:
    q = frame_queues.get(cam_id)
    if q is None:
        q = asyncio.Queue(maxsize=1)   # drop-stale buffer
        frame_queues[cam_id] = q
    return q

async def start_worker_if_needed(cam_id: str):
    """Start a per-camera processing worker if not running."""
    if cam_id in worker_tasks and not worker_tasks[cam_id].done():
        return
    q = ensure_queue(cam_id)
    worker_tasks[cam_id] = asyncio.create_task(process_camera_worker(cam_id, q))

async def stop_worker_if_running(cam_id: str):
    """Cancel and remove the per-camera worker."""
    t = worker_tasks.get(cam_id)
    if t and not t.done():
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    worker_tasks.pop(cam_id, None)
    # keep queue for a bit; it will be reused on next start
    # optional: clear any stale frame
    q = frame_queues.get(cam_id)
    if q:
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except Exception:
                break

async def process_camera_worker(cam_id: str, q: asyncio.Queue):
    """Per-camera consumer: process latest frames -> run model -> score -> broadcast."""
    try:
        while True:
            frame_b64 = await q.get()  # wait for next frame
            try:
                # Decode JPEG -> BGR
                frame_bytes = base64.b64decode(frame_b64)
                jpg_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
                if frame is None:
                    q.task_done()
                    continue

                # 1) Detect bullet holes (x,y,conf) in a worker thread
                det_points = await asyncio.to_thread(model.detect, frame)

                # 2) Score each detection by ring distance to center
                scored_points, raw_score = model.score_points(det_points, frame, cam_id)

                # 3) Baseline: first processed frame after Start becomes baseline
                async with state_lock:
                    if baseline_pending.get(cam_id, False):
                        baseline_score[cam_id] = raw_score
                        baseline_points[cam_id] = scored_points
                        baseline_pending[cam_id] = False

                # 4) Draw markers for visualization
                for p in scored_points:
                    cv2.circle(frame, (int(p["x"]), int(p["y"])), radius=15, color=(0, 0, 255), thickness=3)

                ok, buffer = cv2.imencode(".jpg", frame)
                if not ok:
                    q.task_done()
                    continue
                annotated_b64 = base64.b64encode(buffer).decode("utf-8")

                # 5) Effective score = current - baseline
                async with state_lock:
                    base = baseline_score.get(cam_id, 0)
                effective_score = max(0, raw_score - base)

                # 6) Cache last result for snapshot and refresh
                async with state_lock:
                    last_results[cam_id] = {
                        "score": effective_score,
                        "image": annotated_b64,
                        "points": scored_points,  # x,y,conf,ring,score
                    }

                # 7) Broadcast realtime result
                await broadcast_json({
                    "action": "result",
                    "cam_id": cam_id,
                    "score": effective_score,
                    "image": annotated_b64,
                    "points": scored_points,
                })

            finally:
                q.task_done()
    except asyncio.CancelledError:
        # graceful exit
        return


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.websocket("/ws/rpi")
async def ws_rpi_endpoint(websocket: WebSocket):
    """
    RPi/Ubuntu capture client socket.
    Client -> Server actions:
      - {"action":"register","cameras":[...]}
      - {"action":"frame","cam_id":"cam1","frame":"<base64 jpeg>"}
    Server -> Client actions:
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
                    active_cameras.intersection_update(cams)
                    for c in cams:
                        baseline_pending.setdefault(c, False)
                        ensure_queue(c)  # ensure a queue exists for each known cam
                await broadcast_json(build_snapshot())

            elif action == "frame":
                cam_id = data.get("cam_id")
                frame_b64 = data.get("frame")
                if not (cam_id and frame_b64):
                    continue

                # enqueue latest frame for this camera, dropping stale if needed
                q = ensure_queue(cam_id)
                if q.full():
                    try:
                        _ = q.get_nowait()
                        q.task_done()
                    except Exception:
                        pass
                await q.put(frame_b64)

            # ignore unknown actions

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        # RPi disconnected: clean up
        async with state_lock:
            if rpi_connection is websocket:
                rpi_connection = None
            available_cameras = []
            for cam_id in list(worker_tasks.keys()):
                # cancel all workers
                pass
        # stop workers outside lock
        for cam_id in list(worker_tasks.keys()):
            await stop_worker_if_running(cam_id)

        async with state_lock:
            active_cameras.clear()
            last_results.clear()
            baseline_pending.clear()
            baseline_score.clear()
            baseline_points.clear()
            frame_queues.clear()
        await broadcast_json(build_snapshot())


@app.websocket("/ws/client")
async def ws_client_endpoint(websocket: WebSocket):
    """Admin browser clients connect here to receive snapshots and realtime results."""
    await websocket.accept()
    async with state_lock:
        frontend_connections.add(websocket)
    try:
        await websocket.send_json(build_snapshot())  # hydrate immediately (supports refresh)
    except Exception:
        async with state_lock:
            frontend_connections.discard(websocket)
        return

    try:
        while True:
            await asyncio.sleep(60)  # keepalive
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        async with state_lock:
            frontend_connections.discard(websocket)


@app.get("/start/{cam_id}")
async def start_camera(cam_id: str):
    """Start streaming a given camera and arm baseline for the next processed frame."""
    async with state_lock:
        ws = rpi_connection
        cams = set(available_cameras)
        already_active = cam_id in active_cameras
    if ws is None:
        return {"status": "error", "detail": "Raspberry Pi not connected"}
    if cam_id not in cams:
        return {"status": "error", "detail": f"Camera {cam_id} not available"}

    if already_active:
        return {"status": "ok", "detail": f"{cam_id} already running"}

    try:
        await ws.send_json({"action": "start", "cam_id": cam_id})
        async with state_lock:
            active_cameras.add(cam_id)
            baseline_pending[cam_id] = True   # next processed frame becomes baseline
            baseline_score.pop(cam_id, None)
            baseline_points.pop(cam_id, None)
        # spin up worker for this camera
        await start_worker_if_needed(cam_id)

        await broadcast_json(build_snapshot())
        return {"status": "ok", "detail": f"Start command sent for {cam_id}"}
    except Exception as e:
        return {"status": "error", "detail": f"Failed to send start: {e}"}


@app.get("/stop/{cam_id}")
async def stop_camera(cam_id: str):
    """Stop streaming a given camera (last image/score are preserved for the UI)."""
    async with state_lock:
        ws = rpi_connection
        is_active = cam_id in active_cameras
    if ws is None:
        return {"status": "error", "detail": "Raspberry Pi not connected"}

    if not is_active:
        return {"status": "ok", "detail": f"{cam_id} already stopped"}

    try:
        await ws.send_json({"action": "stop", "cam_id": cam_id})
        async with state_lock:
            active_cameras.discard(cam_id)
        await stop_worker_if_running(cam_id)
        await broadcast_json(build_snapshot())
        return {"status": "ok", "detail": f"Stop command sent for {cam_id}"}
    except Exception as e:
        return {"status": "error", "detail": f"Failed to send stop: {e}"}


@app.get("/stop_all")
async def stop_all():
    """Stop all cameras (debug helper)."""
    async with state_lock:
        ws = rpi_connection
        cams = list(active_cameras)
    if ws is None:
        return {"status": "error", "detail": "Raspberry Pi not connected"}
    try:
        await ws.send_json({"action": "stop_all"})
        # stop all workers
        for cam_id in cams:
            await stop_worker_if_running(cam_id)
        async with state_lock:
            active_cameras.clear()
        await broadcast_json(build_snapshot())
        return {"status": "ok", "detail": "Stop all command sent"}
    except Exception as e:
        return {"status": "error", "detail": f"Failed to send stop_all: {e}"}
