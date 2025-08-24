# main.py (FastAPI backend server)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio, json, base64
import numpy as np
import cv2

app = FastAPI()

# Global state (could also use app.state or dependency injection)
rpi_connection: WebSocket = None       # WebSocket connection to Raspberry Pi
available_cameras: list[str] = []      # list of camera IDs from the Pi
frontend_connections: list[WebSocket] = []  # list of active frontend client websockets

# Placeholder for the model (replace with actual model loading)
class DummyModel:
    def detect(self, image):
        # Dummy detection: returns empty list (no bullet holes)
        # Replace with actual detection logic (e.g., ML model inference)
        return []
model = DummyModel()

def calculate_score(bullet_holes):
    """Calculate score based on bullet hole coordinates. (Dummy implementation)"""
    score = 0
    for (x, y) in bullet_holes:
        # Example logic: closer to center = higher score (you can implement real scoring)
        # Here we'll just give a fixed score per hole for demo.
        score += 10
    return score

@app.get("/")
async def serve_frontend():
    """Serves the admin panel HTML page."""
    # Assume there's an index.html file in the same directory (simple static file serve)
    html_content = ""
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.websocket("/ws/rpi")
async def ws_rpi_endpoint(websocket: WebSocket):
    """WebSocket endpoint for Raspberry Pi client."""
    global rpi_connection, available_cameras
    await websocket.accept()
    rpi_connection = websocket
    try:
        # Listen for messages from Raspberry Pi
        while True:
            msg = await websocket.receive_text()  # receive text frames (JSON from Pi)
            data = json.loads(msg)
            action = data.get("action")
            if action == "register":
                # Pi is registering its cameras
                available_cameras = data.get("cameras", [])
                print(f"RPi connected. Available cameras: {available_cameras}")
                # Notify any connected frontend clients about new cameras
                for client in frontend_connections:
                    await client.send_json({"action": "cameras", "cameras": available_cameras})
            elif action == "frame":
                # Received a frame from a camera
                cam_id = data.get("cam_id")
                frame_b64 = data.get("frame")
                if cam_id and frame_b64:
                    # Decode the base64 image back to bytes
                    frame_bytes = base64.b64decode(frame_b64)
                    # Convert bytes to numpy array and decode to image
                    jpg_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
                    if frame is None:
                        print("Failed to decode image from camera:", cam_id)
                        continue
                    # Run the bullet hole detection model
                    bullet_holes = model.detect(frame)  # list of (x, y) coordinates
                    # Calculate the score for this frame (or cumulative – depending on usage)
                    score = calculate_score(bullet_holes)
                    # Annotate the frame with bullet hole markers (red circles)
                    for (x, y) in bullet_holes:
                        cv2.circle(frame, (x, y), radius=15, color=(0, 0, 255), thickness=3)
                    # Encode the annotated image to JPEG
                    _, buffer = cv2.imencode('.jpg', frame)
                    annotated_b64 = base64.b64encode(buffer).decode('utf-8')
                    # Create result message
                    result_msg = {
                        "action": "result",
                        "cam_id": cam_id,
                        "score": score,
                        "image": annotated_b64
                    }
                    # Send to all connected frontend clients
                    for client in frontend_connections:
                        await client.send_json(result_msg)
            # (Ignore other actions or unknown messages)
    except WebSocketDisconnect:
        # Handle the Pi disconnecting
        print("Raspberry Pi disconnected.")
        rpi_connection = None
        available_cameras = []
        # Notify frontends that cameras are gone/offline (could send an empty list or a special message)
        for client in frontend_connections:
            await client.send_json({"action": "cameras", "cameras": []})
    except Exception as e:
        print("Error in RPi WebSocket handler:", e)
        # If any exception, also treat as disconnect/cleanup
        rpi_connection = None
        available_cameras = []
        for client in frontend_connections:
            await client.send_json({"action": "cameras", "cameras": []})

@app.websocket("/ws/client")
async def ws_client_endpoint(websocket: WebSocket):
    """WebSocket endpoint for admin frontend clients."""
    await websocket.accept()
    frontend_connections.append(websocket)
    # Immediately send current camera list to the new client
    await websocket.send_json({"action": "cameras", "cameras": available_cameras})
    try:
        while True:
            # In this simple design, we don't expect to receive messages from frontend
            # If needed, you could handle incoming messages here.
            await asyncio.sleep(3600)  # keep alive (or handle ping-pong elsewhere)
    except WebSocketDisconnect:
        frontend_connections.remove(websocket)
    except Exception as e:
        print("Error in client WebSocket:", e)
        if websocket in frontend_connections:
            frontend_connections.remove(websocket)

@app.get("/start/{cam_id}")
async def start_camera(cam_id: str):
    """HTTP endpoint to start streaming a given camera."""
    if rpi_connection is None:
        return {"status": "error", "detail": "Raspberry Pi not connected"}
    if cam_id not in available_cameras:
        return {"status": "error", "detail": f"Camera {cam_id} not available"}
    # Send start command to RPi via WebSocket
    cmd = {"action": "start", "cam_id": cam_id}
    await rpi_connection.send_json(cmd)
    return {"status": "ok", "detail": f"Start command sent for {cam_id}"}

@app.get("/stop/{cam_id}")
async def stop_camera(cam_id: str):
    """HTTP endpoint to stop streaming a given camera."""
    if rpi_connection is None:
        return {"status": "error", "detail": "Raspberry Pi not connected"}
    # Send stop command to RPi via WebSocket
    cmd = {"action": "stop", "cam_id": cam_id}
    await rpi_connection.send_json(cmd)
    return {"status": "ok", "detail": f"Stop command sent for {cam_id}"}
