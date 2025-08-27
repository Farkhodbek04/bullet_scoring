# model.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import os
import threading
import numpy as np
import cv2

# Ultralytics YOLO runtime
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(
        "Ultralytics not available. Install with:\n"
        "  pip install ultralytics torch torchvision\n"
        f"Original error: {e}"
    )

# Path to your trained weights. By default looks for ./best.pt
MODEL_WEIGHTS = os.environ.get("BULLETHOLE_WEIGHTS", "best.pt")

# -----------------------------------------------------------------------------
# Per-camera calibration
# -----------------------------------------------------------------------------
# Rings are ordered [r10, r9, r8, r7, r6, r5]
# If you don't provide calibration for a cam_id, the code uses:
#   - center = image center
#   - radii = DEFAULT_RATIOS * (min(image_width, image_height)/2)
DEFAULT_RATIOS = [0.10, 0.18, 0.26, 0.34, 0.42, 0.50]

CALIBRATION: Dict[str, Dict[str, Any]] = {
    # Example per-camera override (uncomment and tune if needed):
    # "cam1": {
    #     "center_px": (960, 540),                         # absolute center (cx, cy)
    #     "radii_px":  [110, 190, 270, 350, 430, 510],     # [r10..r5] in pixels
    # },
    # "cam2": {
    #     "radii_ratio": [0.095, 0.175, 0.255, 0.335, 0.415, 0.495],
    # }
}


class BulletHoleModel:
    def __init__(self, weights_path: str = MODEL_WEIGHTS, conf_thres: float = 0.25):
        self.model = YOLO(weights_path)
        self.conf_thres = conf_thres
        self._predict_lock = threading.Lock()  # make YOLO calls thread-safe

        # Optional warmup to reduce first-frame latency
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            with self._predict_lock:
                _ = self.model.predict(dummy, verbose=False, conf=self.conf_thres)
        except Exception:
            pass

    # -------------------- Detection --------------------
    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on a BGR frame (OpenCV image).
        Returns: [{"x": int, "y": int, "conf": float}, ...]
        Coordinates are pixel centers of predicted bullet holes.
        """
        with self._predict_lock:
            results = self.model.predict(frame_bgr, verbose=False, conf=self.conf_thres)
        if not results:
            return []
        r = results[0]
        dets: List[Dict[str, Any]] = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, conf):
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)
                dets.append({"x": cx, "y": cy, "conf": float(c)})
        return dets

    # -------------------- Scoring --------------------
    def _center_and_radii(self, frame_shape: Tuple[int, int, int], cam_id: Optional[str]) -> Tuple[Tuple[int,int], List[float]]:
        """
        Resolve target center and ring radii for this frame & camera.
        Returns: (center_px (cx,cy), radii_px [r10..r5])
        """
        h, w = frame_shape[:2]
        min_half = min(w, h) / 2.0

        cfg = CALIBRATION.get(cam_id or "", {})
        # Center
        if "center_px" in cfg:
            cx, cy = cfg["center_px"]
        else:
            cx, cy = int(w / 2), int(h / 2)

        # Radii
        if "radii_px" in cfg:
            radii_px = list(cfg["radii_px"])
        else:
            ratios = cfg.get("radii_ratio", DEFAULT_RATIOS)
            radii_px = [float(r * min_half) for r in ratios]

        radii_px = sorted(radii_px)
        return (int(cx), int(cy)), radii_px

    def _ring_for_distance(self, dist: float, radii_px: List[float]) -> int:
        """
        Map radial distance to ring score (10,9,8,7,6,5, else 0).
        radii_px is [r10, r9, r8, r7, r6, r5]
        """
        if dist <= radii_px[0]: return 10
        if dist <= radii_px[1]: return 9
        if dist <= radii_px[2]: return 8
        if dist <= radii_px[3]: return 7
        if dist <= radii_px[4]: return 6
        if dist <= radii_px[5]: return 5
        return 0

    def score_points(
        self,
        points: List[Dict[str, Any]],
        frame_bgr: np.ndarray,
        cam_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        For each detected point, compute its ring and per-shot score based on
        distance to the target center and ring radii.

        Returns:
          (scored_points, total_score)
          where each point adds keys: "ring" (int), "score" (int), "dist" (float)
        """
        (cx, cy), radii_px = self._center_and_radii(frame_bgr.shape, cam_id)
        total = 0
        out: List[Dict[str, Any]] = []
        for p in points:
            x, y = int(p["x"]), int(p["y"])
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            ring = self._ring_for_distance(d, radii_px)
            s = ring
            q = {**p, "ring": ring, "score": int(s), "dist": float(d)}
            out.append(q)
            total += s
        return out, int(total)
